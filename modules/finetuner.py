import os
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import numpy as np

from models.vgg import vgg_16_bn
from models.resnet_cifar10 import resnet_56, resnet_110
from models.resnet_imgnet import resnet_50
from models.googlenet import googlenet
from models.densenet import densenet_40

from collections import OrderedDict
from dataset.dataset_loader import load_data
import utils.common as utils
from modules.bottleneck.module_info import ModuleInfo, ModulesInfo
from utils.arch_modif import prune_layer, get_module_in_model
from utils.model_pruning import prune
from utils.model_loading import load
from modules.attribution.bottleneck import BottleneckReader
from utils.calc_flops import get_flops
from thop import profile

class Finetuner:
    def __init__(self, args, logger=None):
        self.args = args
        self.logger = logger
        self.res = OrderedDict()

        # load training data
        self.input_image_size = {
            "cifar10": 32,        # CIFAR-10
            "imagenet": 224,      # ImageNet
        }[self.args.dataset]

        self.train_loader, self.val_loader = load_data(self.args)

        if self.args.dataset == 'imagenet': #ImageNet
            CLASSES = 1000 #label_smooth: 0.1
            self.criterion = utils.CrossEntropyLabelSmooth(CLASSES, 0.1).cuda()
        else: 
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
        modules = ModulesInfo(self.model, modules_list, self.input_image_size, self.args.device)
        return modules, conv_idx

    def compute_norm(self, layer_name, norm_type='nuclear'):
        """
            Args:
                - layer_name: name of the layer in the model
                - norm_type: norm to apply. Options are 'l1', 'l2', 'nuclear' or 'none'
            Return a list of norms for the filters in the layer layer_name.
        """
        weights = get_module_in_model(self.model, layer_name).weight.data.cpu().detach().numpy()
        if norm_type=='none': return [0.9]*weights.shape[0]
        norm = []
        for i in range(weights.shape[0]):
            weight = weights[i]           # dim (Cin*K*K)
            weight = weight.reshape(weight.shape[0], -1) # dim (Cin)*(K*K)
            if norm_type=='l1':
                norm.append(np.linalg.norm(weight, ord=1)) 
            elif norm_type=='nuclear':
                _, s, _ = np.linalg.svd(weight)
                norm.append(s.sum())
            elif norm_type=='l2':
                norm.append(np.linalg.norm(weight, ord=2))
        # pseudo-normalize
        eps = 0.05
        norm /= (np.max(norm) + eps)
        return norm

    def select_index_flops(self, attribution_score, target_flops, r, layer_to_prune):
        """
            Args:
                - attribution_score: attribution score for each filter in each layer (list of list)
                - target_flops: target flops for the pruned model 
                - r: BottleneckReader
        """
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
        if 'vgg_16' in self.args.arch:
            flops_target = 108.61 * 10**6 if self.args.Mflops_target is None else self.args.Mflops_target * 10**6
            # flops_target = 145.61 * 10**6 if self.args.Mflops_target is None else self.args.Mflops_target * 10**6
            # flops_target = 72.77 * 10**6 if self.args.Mflops_target is None else self.args.Mflops_target * 10**6
        elif 'repvgg_A0' in self.args.arch:
            flops_target = 500. * 10**6  if self.args.Mflops_target is None else self.args.Mflops_target * 10**6
        elif 'repvgg_B0' in self.args.arch:
            flops_target = 900. * 10**6  if self.args.Mflops_target is None else self.args.Mflops_target * 10**6
        elif self.args.arch == 'resnet_56':
            # layers_to_prune = list(range(1,55,2))
            flops_target = 55.84 * 10**6 if self.args.Mflops_target is None else self.args.Mflops_target * 10**6
        elif self.args.arch == 'resnet_110':
            # layers_to_prune = list(range(1,109,2))
            flops_target = 85.3 * 10**6 if self.args.Mflops_target is None else self.args.Mflops_target * 10**6     
        elif self.args.arch == 'densenet_40':
            # flops_target = 167.41 * 10**6 if self.args.Mflops_target is None else self.args.Mflops_target * 10**6
            flops_target = 128.11 * 10**6 if self.args.Mflops_target is None else self.args.Mflops_target * 10**6
        elif self.args.arch == 'googlenet':
            flops_target = 450. * 10**6 if self.args.Mflops_target is None else self.args.Mflops_target * 10**6
        elif self.args.arch == 'resnet_50':
            flops_target = 1700. * 10**6 if self.args.Mflops_target is None else self.args.Mflops_target * 10**6
            # flops_target = 960. * 10**6 if self.args.Mflops_target is None else self.args.Mflops_target * 10**6
        assert(flops_target < maxflops)
        # pruning ratio
        self.logger.info(f"FLOPS pruning ratio is {(1-(flops_target/maxflops)):.2f}")
        
        cnt = 0
        init_lamb = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                if cnt in layers_to_prune:
                    print(f'Add layer: {name}')
                    # compute init lambda with norm
                    norm = self.compute_norm(name, norm_type='none')
                    init_lamb.append(torch.tensor(norm, dtype=torch.float32))
                cnt += 1
        
        self.logger.info("Pruning with information flow")
        reader = BottleneckReader(self.model, 
                                    self.args.arch, 
                                    self.criterion, 
                                    self.logger, modules.modules(), init_lamb, 
                                    layers_to_prune,
                                    conv_idx,
                                    lr=self.args.lr, 
                                    steps=self.args.nb_batches, 
                                    beta=self.args.beta, 
                                    gamma=self.args.gamma, 
                                    target_flops=flops_target,
                                    max_flops=maxflops)
        attribution_score = reader.get_attribution_score(self.train_loader)

        # select the indexes to preserve
        preserved_indexes_all = self.select_index_flops(attribution_score, flops_target, reader, layers_to_prune)

        self.update_results('after_pseudopruning', 'Performances pseudo-pruned model')

        attrib_list_str = "attribution_score[0:12]: \n"
        for j in range(reader.unique_alphas.len()):
            tmp = reader.unique_alphas.get_lambda(j).detach().clone().cpu().numpy()[0:12]
            attrib_list_str += ('[ ' + ' '.join("{:.2f} ".format(lmbd) for lmbd in tmp) + ']\n')
        self.logger.info(attrib_list_str)

        reader.remove_layer()

        # prune
        prune(self.model if len(self.args.device_ids) == 1 else self.model.module, preserved_indexes_all, layers_to_prune, self.args.arch)

        # save pruned model
        filename = os.path.join(self.args.output_dir, 'pruned.pt')
        torch.save({'state_dict': self.model.state_dict(), 'preserved_idx': preserved_indexes_all}, filename)

        self.logger.info(self.model)
        self.update_results('after_pruning', 'Performances pruned model')

    def finetune(self):
        epochs = self.args.epoch_finetuning
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr_finetuning, momentum=self.args.momentum, weight_decay=self.args.wd)
        if self.optimizer_state_dict is not None: optimizer.load_state_dict(self.optimizer_state_dict)
        if self.args.dataset == "imagenet":
            scheduler = utils.CosLR(optimizer, T_max=epochs, len_iter=len(self.train_loader))
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        acc = self.train(optimizer, scheduler, epochs)
        self.update_results('after_finetune', 'Performances finetuned model')

        if self.args.save_plot:
            self.plot_acc(acc, title="Acc vs epoch", filename=f"plot_{self.args.arch}")

    def save_onnx(self):
        # Export the model
        self.model.eval()
        x = torch.randn(1, 3, self.input_image_size, self.input_image_size, requires_grad=True).cuda()
        torch.onnx.export(self.model if len(self.args.device_ids) == 1 else self.model.module,  # model being run
                          x,  # model input (or a tuple for multiple inputs)
                          os.path.join(self.args.output_dir, 'model_best.onnx'),  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=11,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'])  # the model's output names
                        #   dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                        #                 'output': {0: 'batch_size'}})

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
        self.logger.info('-'*40)
        self.logger.info('==> Building model...')
        self.model = eval(self.args.arch)()

        # 2) load weights
        self.logger.info('-'*40)
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


        # for multi-GPU
        if len(self.args.device_ids) > 1:
            self.logger.info('Option: multi-GPU')
            device_ids = []
            for i in range(len(self.args.device_ids)):
                device_ids.append(i)
            self.model = nn.DataParallel(self.model, device_ids=device_ids)

        self.model.to(self.args.device)

        # 3) update results + show various info
        self.logger.info('-'*40)
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



    def update_results(self, key, description):
        """
            Save the current performances of the model.
            Pseudo code:
                new_res = {'measure1_name': measure1_value, 'measure2_name': measure2_value, ..}
                self.res[key] = [description, new_res]
            These results can be dispayed at any time with show_results().
            Args:
                - key (str): the key associated to the new results in the 'res' dictionary.
                - description (str): description of the result (example: 'Performances input model')
        """
        new_res = OrderedDict()
        self.model.to(self.args.device)
        if profile is not None:
            new_res['flops'], new_res['params'] = self.compute_flops_and_params()
        new_res['accuracy'] = self.validate(self.val_loader, self.model, self.criterion, mute=True)[1] # check the accuracy before decomposition
        self.res[key] = [description, new_res]

    def show_results(self, keys=None):
        """
            Display the results which have been saved with.
            Args:
                - keys (str or list): if None, all the results are displayed. Else, display the results at the given keys.
        """
        if isinstance(keys, str): keys = [keys]
        elif keys is None: keys = self.res.keys()

        for key in keys:
            res = self.res[key]
            self.logger.info('-'*40)
            self.logger.info(res[0] + ':') # display description for this key
            for key_res in res[1]:   # display results for this key
                self.logger.info(f' - {key_res}: {res[1][key_res]:,}')

    def plot_acc(self, accuracy, epoch=None, acc_before_compression=None, title="Acc vs epoch", subtitle=None, filename=None, plot_best=False):
        """
            Save a plot of the accuracy evolution.
            Params:
                - accuracy: list of float
                - epoch: list of int, such that accuracy[i] gives the accuracy at epoch epoch[i].
                         If epoch=None, then we assume that accuracy[i] gives the accuracy at epoch i.
                - acc_before_compression: float corresponding to the accuracy before compression
                - title: main title of the graph
                - subtitle: supplementary title (to indicate used param, for instance)
                - filename: name of the png file (do not specify the extension)
        """
        if epoch is None: epoch=np.arange(len(accuracy))
        fig = plt.figure()
        plt.plot(epoch, accuracy)
        if acc_before_compression is not None:
            plt.axhline(y=acc_before_compression, color='r', linestyle='-', label='acc before compression')
        if plot_best:
            plt.axhline(y=np.max(accuracy), color='b', linestyle='-', label='best') # best accuracy
        # if acc_before_compression is not None:
        #     ymin = max(min(np.min(accuracy), acc_before_compression)-1, 0)
        #     ymax = min(max(np.max(accuracy), acc_before_compression)+1, 100)
        ymin = 0
        ymax = 100
        plt.ylim(ymin, ymax)
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        
        plt.title(title + ('' if subtitle is None else ('\n'+subtitle)))
        filepath = f'{self.args.output_dir}/{(title.lower().replace(" ", "_") if filename is None else filename)}.png' 
        fig.savefig(filepath)

    def compute_flops_and_params(self):
        """
            Compute flops and number of parameters for the loaded model
        """
        input_image = torch.randn(1, 3, self.input_image_size, self.input_image_size).cuda()
        if len(self.args.device_ids) > 1:
            flops, params = profile(self.model.module.cuda(), inputs=(input_image,), verbose=False)
        else:
            flops, params = profile(self.model, inputs=(input_image,), verbose=False)
        self.model.to(self.args.device)
        return [int(flops), int(params)]