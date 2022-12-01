import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from modules.bottleneck.bottleneck import Bottleneck, UniqueAlphaMasks
from utils.arch_modif import replace_layer
from utils.calc_flops import get_flops

class BottleneckReader():
    def __init__(self, model, arch, model_criterion, logger, modules: list, init_lamb, layer_to_prune, conv_idx, beta=7, gamma=0.5, steps=50, lr=0.3, batch_size=1, target_flops = None, max_flops=None):
        self.model = model
        self.logger = logger
        self.beta = beta
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = list(model.parameters())[0].device
        self.lr = lr
        self.train_steps = steps

        self.original_layers = []
        self.sequentials = []
        self.conv_info = []
        self.arch = arch
        self.model_criterion = model_criterion

        # create the unique masks (1 for each layer on which we want to prune the output channels):
        self.unique_alphas = UniqueAlphaMasks(self.device)
        for lamb in init_lamb:
            self.unique_alphas.append(lamb)
        self.used_masks = [False for i in range(len(init_lamb))]
        # create Bottlenecks (one for each module in the network):
        self.bottlenecks = []
        cnt = 0
        lambda_in, lambda_out = None, None
        for i, m in enumerate(modules):
            forward_with_mask = None
            
            if self.arch == 'resnet_50':
                if i==0: # initialisation
                    assert(len(layer_to_prune)==53) # we prune all conv layers
                    block_size = 3
                # define lambda_in and lambda_out for m
                lambda_in = lambda_out
                if m.type is nn.Conv2d:
                    if cnt == 0:
                        lambda_out = [cnt]
                        idx_in_block = 0
                    elif cnt in [3,13,26,45]:
                        residual_connection = [cnt]
                        lambda_out = [cnt]
                        idx_in_block = 0
                    elif cnt in [4,14,27,46]: # conv in downsample module (skip connection)
                        lambda_in = [cnt-block_size-1]
                        lambda_out = residual_connection
                    else:
                        idx_in_block = (idx_in_block % block_size) + 1
                        if idx_in_block == 3:
                            lambda_out = residual_connection
                        else:
                            lambda_out = [cnt]
                    cnt+=1
                elif m.type is nn.Linear: lambda_out = None
                else: lambda_out = lambda_in # other modules preserve the number of channels


            elif 'resnet' in self.arch:
                if i==0: # initialisation
                    if self.arch == 'resnet_56': blocks_per_layer = 9
                    elif self.arch == 'resnet_110': blocks_per_layer = 36
                    next_lamb_in = lambda_out
                    next_skip_connection = False
                lambda_in = next_lamb_in
                if m.type is nn.Conv2d:
                    if cnt==0: lambda_out = None
                    else: lambda_out = [cnt]
                    if cnt>1 and cnt%2 == 0:
                        next_skip_connection = True
                    cnt+=1
                elif m.type is nn.Linear: lambda_out = None
                else: lambda_out = lambda_in # other modules preserve the number of channels
                if m.type is nn.ReLU and cnt==1:
                    forward_with_mask = False
                if m.type is nn.ReLU and next_skip_connection:
                    next_skip_connection = False
                    forward_with_mask = False
                if m.type is nn.BatchNorm2d and next_skip_connection:
                    forward_with_mask = True
                    next_lamb_in = None
                else: next_lamb_in = lambda_out

            elif 'vgg_16' in self.arch:
                lambda_in = lambda_out
                if m.type is nn.Conv2d:
                    if conv_idx.index(i) in layer_to_prune:
                        lambda_out = [cnt]
                        cnt+=1
                    else: lambda_out = None
                elif m.type is nn.Linear: lambda_out = None
                else: lambda_out = lambda_in # other modules preserve the number of channels

            elif self.arch == 'densenet_40':
                if i==0: # initialisation
                    assert(len(layer_to_prune)==39) # we prune all conv layers
                    next_lamb_in = None
                    denseblock_size = 12
                lambda_in = next_lamb_in
                if m.type is nn.Conv2d:
                    lambda_out = [cnt]
                    if cnt % (denseblock_size+1) == 0:
                        next_lamb_in = lambda_out
                    else: 
                        next_lamb_in = lambda_in + lambda_out if lambda_in is not None else lambda_out
                    cnt+=1
                elif m.type is nn.Linear: lambda_out = None
                else: lambda_out = lambda_in # other modules preserve the number of channels

            elif self.arch == 'googlenet':
                if i==0: # initialisation
                    assert(len(layer_to_prune)==64) # we prune all conv layers
                    conv_per_inception = 7
                    inception_idx = None
                    next_lamb_in = None
                lambda_in = next_lamb_in
                if m.type is nn.Conv2d:
                    lambda_out = [cnt]
                    next_lamb_in = lambda_out
                    if inception_idx is None: # first conv (not in an inception layer)
                        next_inception_input = lambda_out
                        inception_idx = 0
                    else:
                        inception_idx += 1 
                        if inception_idx==conv_per_inception: inception_idx = 0 
                        if inception_idx in [0,1,3,6]:
                            next_inception_input += lambda_out
                    cnt+=1
                elif m.type is nn.Linear: lambda_out = None
                else: lambda_out = lambda_in # other modules preserve the number of channels

                if m.type is nn.ReLU:
                    if inception_idx == 0: # end of inception block
                        current_inception_input = next_inception_input
                        next_inception_input = []
                    if inception_idx in [0,1,3,6]: next_lamb_in = current_inception_input # end of branch

            if forward_with_mask is None:
                if m.type is nn.ReLU: forward_with_mask = True
                else: forward_with_mask = False


            self.logger.info("-"*5)
            self.logger.info(f'module: {m.module}')
            self.logger.info(f'cin: {lambda_in}')
            self.logger.info(f'cout: {lambda_out}')
            self.logger.info('active' if forward_with_mask else 'inactive')

            # generate Bottleneck
            b = Bottleneck(m.type, m.conv_info, lambda_in, lambda_out, self.unique_alphas, forward_with_mask)
            self.bottlenecks.append(b)
            self.original_layers.append(m.module)
            self.sequentials.append(nn.Sequential(m.module, b))
            #
            if lambda_in is not None:
                for i in lambda_in: self.used_masks[i] = True
            if lambda_out is not None:
                for i in lambda_out: self.used_masks[i] = True
        self.logger.info(f"Used masks: {[i for i in range(len(self.used_masks)) if self.used_masks[i]]}")
        self.logger.info(f"{len(self.used_masks)} unique masks in total")
        self.layer_to_prune = layer_to_prune

        for i in range(len(self.used_masks)):
            if not self.used_masks[i]:
                self.unique_alphas.set_lambda(torch.ones(init_lamb[i].size(0)), i)
        for b in self.bottlenecks:
            b.update_lambdas()

        # init loss:
        flops = self.compute_flops(ignore_mask=True)

        self.base_flops = max_flops-flops
        print(f"Base flops: {int(self.base_flops):,}")
        self.target_flops = target_flops
        print(f"Target flops: {int(self.target_flops):,}")
        self.max_flops = max_flops
        print(f"Max flops: {int(self.max_flops):,}")

    def compute_flops(self, ignore_mask=False):
        flops = 0
        for b in self.bottlenecks:
            flops += b.compute_flops(ignore_mask=ignore_mask)
        return flops
        

    def calc_loss_terms(self):
        """ Calculate the loss terms """

        flops = self.base_flops + self.compute_flops()
        
        # pruning_loss = abs(self.target_flops - flops) / self.target_flops
        if flops > self.target_flops:
            pruning_loss = (flops-self.target_flops) / (self.max_flops-self.target_flops)
        else:
            pruning_loss = 1-(flops/self.target_flops)
        print(f'total flops: {int(flops):,}')

        bool_loss = 0
        for i in range(self.unique_alphas.len()):
            if self.used_masks[i]:
                lamb = self.unique_alphas.get_lambda(i)
                nb_filters = lamb.size(0)
                bool_loss += (torch.sum(torch.abs(lamb-torch.round(lamb)))/nb_filters)
        bool_loss /= sum(self.used_masks)
        
        return pruning_loss, bool_loss

    def calc_loss(self, inputs, targets, step, maxstep, verbose=False):

        for b in self.bottlenecks:
            b.update_lambdas()
        ce_loss_total  = 0
        for j in range(inputs.size(0)): #for each single image
            img = inputs[j].unsqueeze(0)
            batch = img.expand(self.batch_size, -1, -1, -1), targets[j].expand(self.batch_size)
            out = self.model(batch[0]) #forward-pass using different noises with self.batch_size
            ce_loss_total += self.model_criterion(out, batch[1])/inputs.size(0)
        pruning_loss_total, bool_loss_total = self.calc_loss_terms()
        loss = ce_loss_total + self.beta * pruning_loss_total + self.gamma * bool_loss_total
        if verbose:
            self.logger.info(f"loss = {ce_loss_total:.3f} + {self.beta * pruning_loss_total:.3f} + {self.gamma * bool_loss_total:.3f} = {loss:.3f}")
        return loss

    def get_attribution_score(self, dataloader):
        # self._run_training(input_t, target)
        self._run_training(dataloader)
        return self.best_attribution_score

    def _run_training(self, dataloader):
        # Attach layer and train the bottleneck
        print(f'nb unique bottlenecks: {sum(self.used_masks)}')
        for i in range(len(self.bottlenecks)):
            replace_layer(self.model, self.original_layers[i], self.sequentials[i])
        self._train_bottleneck(dataloader)
        
    def remove_layer(self):
        for i in range(len(self.bottlenecks)):
            replace_layer(self.model, self.sequentials[i], self.original_layers[i])

    def update_alpha_with(self, init_lamb_list):
        for i, init_lamb in enumerate(init_lamb_list):
            self.unique_alphas.set_lambda(init_lamb, i)
        for b in self.bottlenecks:
            b.update_lambdas()

    def normalised_kendall_tau_distance(self, values1, values2): #refer to Wiki (https://en.wikipedia.org/wiki/Kendall_tau_distance)
        # FOR REBUTTAL EXPERIMENTS
        n = len(values1)
        assert len(values2) == n, "Both lists have to be of equal length"
        i, j = np.meshgrid(np.arange(n), np.arange(n))
        a = values1
        b = values2
        ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]), np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
        return ndisordered / (n * (n - 1))

    def _train_bottleneck(self, dataloader):
        params = self.unique_alphas.get_params(self.used_masks)
        optimizer = torch.optim.Adam(lr=self.lr, params=params)
        self.best_attribution_score = []
        self.model.eval()
        optimizer.zero_grad()
        best_loss = 999999
        best_epoch = 1
        i = 0
        if self.arch == 'resnet_50': accumulation_steps = 1
        else: accumulation_steps = 1

        # ==== modif rebuttal
        compute_dissim = False
        if compute_dissim:
            lmd_prev = None # for rebuttal
            dir_ = 'densenet'
            f_avg = open(f"result/rebuttal/dissimilarity/{dir_}/dissim_avg.txt", "w") # for rebuttal
            observed_layers = [0, 3, 6, 12, 50, 80]
            layer_files = []
            for l in observed_layers:
                layer_files.append(open(f"result/rebuttal/dissimilarity/{dir_}/dissim_l{l}.txt", "w")) # for rebuttal
        # === end modif rebuttal

        while i < self.train_steps:
            for (inputs, targets) in dataloader:
                if i == self.train_steps: break
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                if (i+1) % accumulation_steps == 0:
                    self.logger.info(f"=== Batch {int((i+1)/accumulation_steps)}/{int(self.train_steps/accumulation_steps)}")
                
                loss = self.calc_loss(inputs, targets, i, self.train_steps, verbose=True)/accumulation_steps
                loss.backward()
                
                if (i+1) % accumulation_steps == 0:
                    if loss < best_loss and i>0.6*self.train_steps:
                        best_epoch = i+1
                        best_loss = float(loss.detach().clone().cpu())
                        self.best_attribution_score = []
                        for j in range(self.unique_alphas.len()):
                            lamb = self.unique_alphas.get_lambda(j)
                            self.best_attribution_score.append(lamb.detach().clone().cpu().tolist())
                    optimizer.step()
                    optimizer.zero_grad()
                    # show params
                    accumulated_step = ((i+1) / accumulation_steps)
                    if accumulated_step < 20 or accumulated_step%100==0: # after 20 epochs, values don't change that much
                        attrib_list_str = "attribution_score[0:12]: " + ('' if sum(self.used_masks)==1 else '\n')
                        for j in range(self.unique_alphas.len()):
                            lmd_layer = self.unique_alphas.get_lambda(j).detach().clone().cpu().numpy()
                            attrib_list_str += ('[ ' + ' '.join("{:.2f} ".format(lmbd) for lmbd in lmd_layer[0:12]) + ']\n')
                        self.logger.info(attrib_list_str)

                    # ==== modif rebuttal
                    # python3 main.py --output_dir result/rebuttal/dissimilarity/test --mode prune --arch vgg_16_bn
                    # python3 run.py --output_dir result/rebuttal/nb_epoch --mode prune --arch vgg_16_bn
                    if compute_dissim:
                        lmd_all = []
                        for j in range(self.unique_alphas.len()):
                            lmd_layer = self.unique_alphas.get_lambda(j).detach().clone().cpu().numpy()
                            lmd_all.append(lmd_layer)
                            # lmd_all.append(np.argsort(lmd_layer))

                        if lmd_prev is not None:
                            ranking_dissimilarity = 0
                            for k in range(self.unique_alphas.len()):
                                layer_ranking_dissim = self.normalised_kendall_tau_distance(lmd_all[k], lmd_prev[k])
                                ranking_dissimilarity += layer_ranking_dissim
                                if k in observed_layers:
                                    layer_files[observed_layers.index(k)].write(f"{i} {layer_ranking_dissim:.4f}" + '\n')
                                    
                                # ranking_dissimilarity += st.kendalltau(lmd_all[k], lmd_prev[k])[0]
                            ranking_dissimilarity /= self.unique_alphas.len()
                            f_avg.write(f"{i} {ranking_dissimilarity:.4f}" + '\n')
                            print(f"ranking_dissimilarity = {ranking_dissimilarity}")
                        else: import scipy.stats as st
                        lmd_prev = lmd_all
                    # === end modif rebuttal

                    print()
                i+=1
        # ==== modif rebuttal
        if compute_dissim:
            f_avg.close() # for rebuttal
            for f in layer_files: f.close()
        # === end modif rebuttal
        self.logger.info(f'===\nBest loss was {best_loss:.2f} at iteration {best_epoch}\n')