import torch.nn as nn
import torch
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
        self.layer_to_prune = layer_to_prune
        self.original_layers = []
        self.sequentials = []
        self.arch = arch
        self.model_criterion = model_criterion

        # create the unique masks (1 for each layer on which we want to prune the output channels):
        self.unique_alphas = UniqueAlphaMasks(self.device)
        for lamb in init_lamb:
            self.unique_alphas.append(lamb) #trainable parameters = alpha
        self.used_masks = [False for _ in range(len(init_lamb))]
        # create Bottlenecks (one for each module in the network):
        self.bottlenecks = []
        cnt = 0
        lambda_in, lambda_out = None, None
        for i, m in enumerate(modules):
            forward_with_mask = None
            
            if 'resnet_56' in self.arch:
                if i==0: # initialisation
                    next_lamb_in = lambda_out
                    next_skip_connection = False
                lambda_in = next_lamb_in
                if m.type is nn.Conv2d:
                    if cnt==0: lambda_out = None
                    else: lambda_out = [cnt]
                    if cnt>1 and cnt%2 == 0:
                        next_skip_connection = True
                    cnt+=1
                elif m.type is nn.Linear: 
                    lambda_out = None
                    print(f'Linear')
                else: lambda_out = lambda_in # other modules preserve the number of channels
                if m.type is nn.ReLU and cnt==1:
                    forward_with_mask = False
                if m.type is nn.ReLU and next_skip_connection:
                    next_skip_connection = False
                    forward_with_mask = False
                if m.type is nn.BatchNorm2d and next_skip_connection:
                    forward_with_mask = True #ReLU 뒤에는 Conv 한번 더 나옴
                    next_lamb_in = None
                else: next_lamb_in = lambda_out

            if forward_with_mask is None:
                forward_with_mask = m.type is nn.ReLU

            self.logger.info("-"*5)
            self.logger.info(f'module: {m.module}')
            self.logger.info(f'cin: {lambda_in}')
            self.logger.info(f'cout: {lambda_out}')
            self.logger.info('active' if forward_with_mask else 'inactive')

            # generate Bottleneck
            b = Bottleneck(m.type, m.conv_info, lambda_in, lambda_out, self.unique_alphas, forward_with_mask)
            self.bottlenecks.append(b) #Bottleneck module
            self.original_layers.append(m.module) #Original module
            self.sequentials.append(nn.Sequential(m.module, b)) #Original + Bottleneck modules (Conv 뒤에 Bottleneck이 붙은 module (이후에 교체됨))
            #
            if lambda_in is not None:
                for i in lambda_in: self.used_masks[i] = True
            if lambda_out is not None:
                for i in lambda_out: self.used_masks[i] = True
        self.logger.info(f"Used masks: {[i for i in range(len(self.used_masks)) if self.used_masks[i]]}")
        self.logger.info(f"{len(self.used_masks)} unique masks in total")


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
        flops = self.base_flops + self.compute_flops()
        
        # pruning_loss = abs(self.target_flops - flops) / self.target_flops
        if flops > self.target_flops:
            pruning_loss = (flops-self.target_flops) / (self.max_flops-self.target_flops)
        else:
            pruning_loss = 1-(flops/self.target_flops)
        print(f'total flops: {int(flops):,}')
        
        return pruning_loss

    def calc_loss(self, inputs, targets, step, maxstep, verbose=False):
        for b in self.bottlenecks:
            b.update_lambdas()
        out = self.model(inputs)
        ce_loss_total = self.model_criterion(out, targets)
        pruning_loss_total = self.calc_loss_terms()
        loss = ce_loss_total + self.beta * pruning_loss_total #loss CE loss, pruning target FLOP loss
        if verbose:
            self.logger.info(f"loss = {ce_loss_total:.3f} + {self.beta * pruning_loss_total:.3f} = {loss:.3f}")
        return loss

    def get_attribution_score(self, dataloader):
        # self._run_training(input_t, target)
        self._run_training(dataloader)
        return self.best_attribution_score

    def _run_training(self, dataloader):
        # Attach layer and train the bottleneck
        print(f'nb unique bottlenecks: {sum(self.used_masks)}')
        for i in range(len(self.bottlenecks)):
            replace_layer(self.model, self.original_layers[i], self.sequentials[i]) #bottleneck 싹다 넣어주고
        self._train_bottleneck(dataloader) #training 시작
        
    def remove_layer(self):
        for i in range(len(self.bottlenecks)):
            replace_layer(self.model, self.sequentials[i], self.original_layers[i])

    def update_alpha_with(self, init_lamb_list):
        for i, init_lamb in enumerate(init_lamb_list):
            self.unique_alphas.set_lambda(init_lamb, i)
        for b in self.bottlenecks:
            b.update_lambdas()

    def _train_bottleneck(self, dataloader):
        params = self.unique_alphas.get_params(self.used_masks)
        optimizer = torch.optim.Adam(lr=self.lr, params=params)
        self.best_attribution_score = []
        self.model.eval() #weight가 freezing 되나 봄
        optimizer.zero_grad()
        best_loss = 999999
        best_epoch = 1
        i = 0
        accumulation_steps = 1

        while i < self.train_steps:
            for (inputs, targets) in dataloader:
                if i == self.train_steps: break
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.logger.info(f"=== Batch {int((i+1))}/{int(self.train_steps)}")
                
                loss = self.calc_loss(inputs, targets, i, self.train_steps, verbose=True)
                loss.backward()
                
                if loss < best_loss and i>0.6*self.train_steps:
                    best_epoch = i+1
                    best_loss = float(loss.detach().clone().cpu())
                    self.best_attribution_score = []
                    for j in range(self.unique_alphas.len()):
                        lamb = self.unique_alphas.get_lambda(j)
                        self.best_attribution_score.append(lamb.detach().clone().cpu().tolist()) #self.best_attribution_score에 best pruning ratio가 저장
                optimizer.step()
                optimizer.zero_grad()
                i+=1

        self.logger.info(f'===\nBest loss was {best_loss:.2f} at iteration {best_epoch}\n')