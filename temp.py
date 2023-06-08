import os
import time, datetime
import utils.common as utils
from utils.parser import get_args

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np

import matplotlib.pyplot as plt

from models.resnet_cifar10 import resnet_56

from collections import OrderedDict
from dataset.dataset_loader import load_data
import utils.common as utils
from modules.bottleneck.module_info import ModuleInfo, ModulesInfo
from utils.arch_modif import prune_layer, get_module_in_model
from utils.model_loading import load
from modules.attribution.bottleneck import BottleneckReader
from utils.calc_flops import get_flops
from thop import profile


args = get_args()

# Data Acquisition
args.print_freq = (256*50)//args.batch_size  # CIFAR-10
args.num_classes = 10

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.device_ids = list(map(int, args.gpu.split(',')))

torch.manual_seed(args.seed)  ##for cpu
torch.cuda.manual_seed(args.seed)  ##for gpu

if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

utils.record_config(args)
now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
logger = utils.get_logger(os.path.join(args.output_dir, 'logger'+now+'.log'))

def main():
    cudnn.benchmark = True
    cudnn.enabled=True

    tuner = Finetuner(args, logger) #import the network in a predefined class
    tuner.prune_model()

class Finetuner:
    def __init__(self, args, logger=None):
        self.args = args
        self.logger = logger
        self.res = OrderedDict()

        # load training data
        self.input_image_size = 32 # CIFAR-10
        self.train_loader, self.val_loader = load_data(self.args)

        self.criterion = nn.CrossEntropyLoss().cuda()
        self.load_model()

    #================================ MAIN CODE

    def get_modules(self):
        """
            Return a list containing info for each module in self.model (input & output size, nb channels, etc),
            as well as the index of the convs relatively to all the modules in the model.
        """
        self.logger.info("Save modules info...")
        modules_list=[]
        conv_idx=[]
        cnt = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU, nn.BatchNorm2d, 
                                    nn.MaxPool2d, nn.AdaptiveMaxPool2d, 
                                    nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                modules_list.append(ModuleInfo(module))
                if isinstance(module, nn.Conv2d):
                    conv_idx.append(cnt)
                cnt += 1
        modules = ModulesInfo(self.model, modules_list, self.input_image_size, self.args.device) #forward hook 써서 각 module의 h,w,c_in,c_out 등의 정보를 얻음
        return modules, conv_idx

    def compute_norm(self, layer_name, norm_type='nuclear'):
        weights = get_module_in_model(self.model, layer_name).weight.data.cpu().detach().numpy() #weight를 얻고
        return [0.9]*weights.shape[0] #normal은 여기 통과 (output channel 갯수 만큼)

    def select_index_flops(self, attribution_score, target_flops, r, layer_to_prune):
        with torch.no_grad():
            # 1. we find a threshold to have the right number of flops, using dychotomy
            self.logger.info('Looking for optimal threshold...')
            
            thres = 0.5
            delta = 0.25

            attrib = [[1 if e>thres else 0 for e in l] for l in attribution_score]
            base_flops = r.base_flops
            flops = base_flops
            iteration = 0
            while abs(flops-target_flops)>50000 and iteration<50:
                self.logger.info(f'Testing threshold {thres}')
                attrib = [[1 if e>thres else 0 for e in l] for l in attribution_score]
                # make sure that nothing is 100% pruned
                for i in range(len(attrib)):
                    if sum(attrib[i])==0:
                        attrib[i][np.argmax(attribution_score[i])] = 1

                # pseudo-prune model with attrib
                r.update_alpha_with(attrib)
                flops = base_flops + r.compute_flops()

                self.logger.info(f'Distance to target: {int(abs(flops-target_flops)):,}')
                if flops > target_flops: thres += delta
                else: thres -= delta
                delta /= 2
                iteration +=1
            # 2. once we found the right threshold, we select the indexes to prune
            from itertools import groupby
            preserved_indexes_all = [[bool(e) for e in l] for l in attrib]
            preserved_indexes_all = [[j,i] for j in range(len(preserved_indexes_all)) for i in range(len(preserved_indexes_all[j])) if preserved_indexes_all[j][i]]
            preserved_indexes_all = [[i[1] for i in e] for _,e in groupby(preserved_indexes_all, lambda x: x[0])]

            return preserved_indexes_all

    def prune_model(self):
        # generate modules info
        modules, conv_idx = self.get_modules()

        # max flops in the model
        maxflops, _ = self.compute_flops_and_params()
        # index of layers to prune and target FLOPS (for each architecture)
        layers_to_prune = list(np.arange(len(conv_idx)))
        # layers_to_prune = list(range(1,55,2))
        flops_target = 55.84 * 10**6 if self.args.Mflops_target is None else self.args.Mflops_target * 10**6
        assert(flops_target < maxflops)
        # pruning ratio
        self.logger.info(f"FLOPS pruning ratio is {(1-(flops_target/maxflops)):.2f}")
        
        cnt = 0
        init_lamb = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d): #bottleneck 낄 위치 (즉, conv 자리 찾기 & 해당 크기만큼 tensor 생성해서 initialization)
                if cnt in layers_to_prune:
                    print(f'Add layer: {name}')
                    # compute init lambda with norm
                    norm = self.compute_norm(name, norm_type='none')
                    init_lamb.append(torch.tensor(norm, dtype=torch.float32)) #init_lamb = lambda 초기값 (이 값이 결국 pruning 할지 말지를 결정)
                cnt += 1
        
        self.logger.info("Pruning with information flow")
        reader = BottleneckReader(self.model, 
                                    self.args.arch, 
                                    self.criterion, 
                                    self.logger, modules.modules(), init_lamb, 
                                    layers_to_prune,
                                    conv_idx, #conv index에 대한 정보 (이 부분만 pruning 진행할꺼니까)
                                    lr=self.args.lr, 
                                    steps=self.args.nb_batches, 
                                    beta=self.args.beta, 
                                    gamma=self.args.gamma, 
                                    target_flops=flops_target,
                                    max_flops=maxflops) #조건을 넣어주고 (바로 사용되지는 않지만), injected bottleneck 심어줌
        attribution_score = reader.get_attribution_score(self.train_loader) #여러번의 iteration을 통해, 전체 block (= layer) 내 각각의 filters in blocks 만큼 bottleneck score 계산

        # select the indexes to preserve
        preserved_indexes_all = self.select_index_flops(attribution_score, flops_target, reader, layers_to_prune)

        self.update_results('after_pseudopruning', 'Performances pseudo-pruned model')

        attrib_list_str = "attribution_score[0:12]: \n"
        for j in range(reader.unique_alphas.len()):
            tmp = reader.unique_alphas.get_lambda(j).detach().clone().cpu().numpy()[0:12]
            attrib_list_str += ('[ ' + ' '.join("{:.2f} ".format(lmbd) for lmbd in tmp) + ']\n')
        self.logger.info(attrib_list_str)

        reader.remove_layer()

    #================================ LOAD/TRAIN/VALIDATE
    def load_model(self):
        """
            Loading in 3 steps:
            1) generate the architecture
            2) load the weights
            3) show/save perf after loading
        """
        checkpoint = torch.load(self.args.loaded_model_path, map_location=self.args.device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        self.start_epoch = int(checkpoint['epoch']) if 'epoch' in checkpoint.keys() and self.args.resume else 0
        self.optimizer_state_dict = checkpoint['optimizer'] if 'optimizer' in checkpoint.keys() and self.args.resume else None
        self.preserved_idx = checkpoint['preserved_idx'] if 'preserved_idx' in checkpoint.keys() else None

        # 1) generate arch
        self.logger.info('==> Building model...')
        self.model = eval(self.args.arch)()

        # 2) load weights
        self.logger.info('==> Loading weights into the model...')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace('module.', '')
            if self.args.arch in ['densenet_40']:
                new_key = new_key.replace('linear.', 'fc.')
            new_state_dict[new_key] = v
        state_dict = new_state_dict

        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError:
            # if loading doesn't work, we assume the loaded model is pruned
            load(self.model, state_dict, self.args.arch, self.preserved_idx)

        self.model.to(self.args.device)

        # 3) update results + show various info
        self.logger.info(self.model) # display model arch

        self.update_results('loading', 'Performances input model')

    def train(self, optimizer, scheduler, total_epochs):
        if total_epochs ==0: return
        def train_one_epoch(epoch, opti, sched):
            batch_time = utils.AverageMeter('Time', ':6.3f')
            data_time = utils.AverageMeter('Data', ':6.3f')
            losses = utils.AverageMeter('Loss', ':.4e')
            top1 = utils.AverageMeter('Acc@1', ':6.2f')
            top5 = utils.AverageMeter('Acc@5', ':6.2f')

            self.model.train()
            end = time.time()

            cur_lr = opti.param_groups[0]['lr']

            num_iter = len(self.train_loader)
            self.logger.info(f'Epoch[{epoch}]')
            self.logger.info('learning_rate: ' + str(cur_lr))
            for i, (images, target) in enumerate(self.train_loader):
                data_time.update(time.time() - end)
                images = images.to(self.args.device)
                target = target.to(self.args.device)

                # compute output
                logits = self.model(images)
                loss = self.criterion(logits, target)

                # measure accuracy and record loss
                if self.args.num_classes > 5: prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                else: prec1 = utils.accuracy(logits, target, topk=(1, ))
                n = images.size(0)
                losses.update(loss.item(), n)   #accumulated loss
                top1.update(prec1.item(), n)
                if self.args.num_classes > 5:
                    top5.update(prec5.item(), n)

                # compute gradient and do SGD step
                opti.zero_grad()
                loss.backward()
                opti.step()
                if isinstance(sched, (torch.optim.lr_scheduler.OneCycleLR, utils.CosLR)):
                    sched.step() # for OneCycleLR, we need to compute a scheduler step at each batch

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.args.print_freq == 0:
                    self.logger.info(
                        '  ({0}/{1}):'
                        'Loss {loss.avg:.4f} '
                        'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                            i, num_iter, loss=losses,
                            top1=top1, top5=top5))

            if not isinstance(sched, (torch.optim.lr_scheduler.OneCycleLR, utils.CosLR)):
                sched.step()
            return losses.avg, top1.avg, top5.avg

        best_top1_acc = 0
        best_top5_acc = 0
        epoch_best_top1_acc = 0
        # adjust the learning rate according to the checkpoint
        for epoch in range(self.start_epoch):
            if isinstance(scheduler, (torch.optim.lr_scheduler.OneCycleLR, utils.CosLR)):
                for i in range(len(self.train_loader)): scheduler.step()
            else: scheduler.step()

        # train the model
        epoch = self.start_epoch
        acc = []
        while epoch < total_epochs:
            train_obj, train_top1_acc, train_top5_acc = train_one_epoch(epoch, optimizer, scheduler)
            valid_obj, valid_top1_acc, valid_top5_acc = self.validate(self.val_loader, self.model, self.criterion)
            if self.args.save_plot:
                acc.append(valid_top1_acc.cpu())
            is_best = False
            if valid_top1_acc > best_top1_acc:
                best_top1_acc = valid_top1_acc
                epoch_best_top1_acc = epoch
                is_best = True

            utils.save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'best_top1_acc': best_top1_acc,
                'optimizer': optimizer.state_dict(),
                'preserved_idx': self.preserved_idx
            }, is_best, self.args.output_dir)

            epoch += 1
            self.logger.info("=>Best accuracy {:.3f} (at epoch {})".format(best_top1_acc, epoch_best_top1_acc)) 
            self.logger.info("-"*4)

        #recall the best model
        best_model = os.path.join(self.args.output_dir, 'model_best.pt')
        checkpoint = torch.load(best_model, map_location=self.args.device)
        state_dict = checkpoint['state_dict']
        self.model.load_state_dict(state_dict)
        return acc

    def validate(self, val_loader, model, criterion, mute=False):
        batch_time = utils.AverageMeter('Time', ':6.3f')
        losses = utils.AverageMeter('Loss', ':.4e')
        top1 = utils.AverageMeter('Acc@1', ':6.2f')
        top5 = utils.AverageMeter('Acc@5', ':6.2f')

        # switch to evaluation mode
        model.eval()
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(val_loader):
                images = images.to(self.args.device)
                target = target.to(self.args.device)

                # compute output
                logits = model(images)
                loss = criterion(logits, target)

                # measure accuracy and record loss
                pred1, pred5 = utils.accuracy(logits, target, topk=(1, 5))
                n = images.size(0)
                losses.update(loss.item(), n)
                top1.update(pred1[0], n)
                top5.update(pred5[0], n)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

            if not mute:
                self.logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                    .format(top1=top1, top5=top5))

        return losses.avg, top1.avg, top5.avg
    
    #================================ SHOW PERFORMANCES
    def update_results(self, key, description): #Save the current performances of the model.
        new_res = OrderedDict()
        self.model.to(self.args.device)
        if profile is not None:
            new_res['flops'], new_res['params'] = self.compute_flops_and_params()
        new_res['accuracy'] = self.validate(self.val_loader, self.model, self.criterion, mute=True)[1] # check the accuracy before decomposition
        self.res[key] = [description, new_res]

    def show_results(self, keys=None): #Display the results which have been saved with.        
        if isinstance(keys, str): keys = [keys]
        elif keys is None: keys = self.res.keys()

        for key in keys:
            res = self.res[key]
            self.logger.info('-'*40)
            self.logger.info(res[0] + ':') # display description for this key
            for key_res in res[1]:   # display results for this key
                self.logger.info(f' - {key_res}: {res[1][key_res]:,}')

    def compute_flops_and_params(self): #Compute flops and number of parameters for the loaded model
        input_image = torch.randn(1, 3, self.input_image_size, self.input_image_size).cuda()
        flops, params = profile(self.model, inputs=(input_image,), verbose=False)
        self.model.to(self.args.device)
        return [int(flops), int(params)]

if __name__ == '__main__':
    main()