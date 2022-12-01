import torch
import torch.nn as nn
import numpy as np

from models.googlenet import Inception

import utils.common as utils
from utils.arch_modif import prune_layer, get_module_in_model

def prune(model, preserved_indexes_all, layers_to_prune, arch):
    if 'vgg_16' in arch:
        prune_vgg(model, preserved_indexes_all, layers_to_prune)
    elif arch == 'resnet_50':
        prune_resnet50(model, preserved_indexes_all, layers_to_prune)
    elif 'resnet' in arch:
        prune_resnet(model, preserved_indexes_all, layers_to_prune)
    elif arch=='densenet_40':
        prune_densenet40(model, preserved_indexes_all, layers_to_prune)
    elif arch=='googlenet':
        prune_googlenet(model, preserved_indexes_all, layers_to_prune)

# --------------------------------------------------------------------------------

def prune_vgg(model, preserved_indexes_all, layers_to_prune):
    """
    Method to prune vgg.
    """
    previous_pruned = False
    cnt = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if previous_pruned:
                # prune input channel
                prune_layer(model, name, preserved_indexes, 1)
                previous_pruned = False
            if cnt in layers_to_prune:
                idx = layers_to_prune.index(cnt)
                preserved_indexes = preserved_indexes_all[idx]
                prune_ratio = prune_layer(model, name, preserved_indexes, 0)
                print(f'[{cnt}] Target: {name} ({int(prune_ratio*100)}%)')
                previous_pruned = True
            cnt+=1
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)) and previous_pruned:
            prune_layer(model, name, preserved_indexes) 

# --------------------------------------------------------------------------------

def prune_resnet(model, preserved_indexes_all, layers_to_prune):
    """
    Method to prune Resnet (20, 32, 44, 56, 110, 1202).
    """
    previous_pruned = False
    cnt = 0

    for name_block, block in model.named_children():
        if isinstance(block, nn.Conv2d): cnt+=1
        # === LAYER ===
        if isinstance(block, nn.Sequential):
            # ==== BOTTLENECK IN LAYER ===
            for name_basicblock, basicblock in block.named_children():
                fullname_basicblock = f"{name_block}.{name_basicblock}"
                local_cnt = 0
                for name_module, module in basicblock.named_modules():
                    name = f"{fullname_basicblock}.{name_module}"
                    if isinstance(module, nn.Conv2d):
                        if previous_pruned:
                            # prune input channel
                            if local_cnt!=0: prune_layer(model, name, preserved_indexes, 1)
                            previous_pruned = False
                        if cnt in layers_to_prune:
                            idx = layers_to_prune.index(cnt)
                            preserved_indexes = preserved_indexes_all[idx]
                            prune_ratio = prune_layer(model, name, preserved_indexes, 0)
                            print(f'[{cnt}] Target: {name} ({int(prune_ratio*100)}%)')
                            previous_pruned = True
                            if local_cnt==1:
                                m = model
                                for m_name in fullname_basicblock.split("."):
                                    m = m[int(m_name)] if m_name.isdigit() else getattr(m, m_name)
                                setattr(m, 'out_index', torch.tensor(preserved_indexes))
                        cnt+=1
                        local_cnt+=1
                    elif isinstance(module, nn.BatchNorm2d) and previous_pruned:
                        prune_layer(model, name, preserved_indexes)

# def prune_resnet(model, preserved_indexes_all, layers_to_prune):
#     """
#     Method to prune Resnet (20, 32, 44, 56, 110, 1202).
#     This version only prunes the layers without skip connection!
#     """
#     previous_pruned = False
#     cnt = 0
#     for name, module in model.named_modules():
#         if isinstance(module, (nn.Conv2d, nn.Linear)):
#             if previous_pruned:
#                 # prune input channel
#                 prune_layer(model, name, preserved_indexes, 1)
#                 previous_pruned = False
#             if cnt in layers_to_prune:
#                 print(f'[{cnt}] Target: {name}')
#                 idx = layers_to_prune.index(cnt)
#                 preserved_indexes = preserved_indexes_all[idx]
#                 prune_layer(model, name, preserved_indexes, 0)
#                 previous_pruned = True
#             cnt+=1
#         elif (isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d)) and previous_pruned:
#             prune_layer(model, name, preserved_indexes)    

# --------------------------------------------------------------------------------

def prune_resnet50(model, preserved_indexes_all, layers_to_prune):
    """
    Method to prune Resnet50.
    """
    previous_pruned = False
    cnt = 0

    for name_block, block in model.named_children():
        if isinstance(block, nn.Conv2d):
            if cnt in layers_to_prune:
                idx = layers_to_prune.index(cnt)
                preserved_indexes = preserved_indexes_all[idx]
                prune_layer(model, name_block, preserved_indexes, 0)
                previous_pruned = True
            cnt+=1
        if isinstance(block, nn.BatchNorm2d):
            prune_layer(model, name_block, preserved_indexes)
        # === LAYER ===
        if isinstance(block, nn.ModuleList):
            # ==== BOTTLENECK IN LAYER ===
            for name_bottleneck, bottleneck in block.named_children():
                idx_in_bottleneck = 1
                preserved_indexes_local = preserved_indexes
                previous_pruned_local = previous_pruned
                # ==== MODULES IN BOTTLENECK ===
                for name_module, module in bottleneck.named_modules():
                    name = f"{name_block}.{name_bottleneck}.{name_module}"
                    if isinstance(module, nn.Conv2d):
                        if idx_in_bottleneck == 4: # downsample
                            if previous_pruned: prune_layer(model, name, preserved_indexes, 1)
                            if previous_pruned_local: 
                                prune_ratio = prune_layer(model, name, preserved_indexes_local, 0)
                                print(f'[{cnt}] Target: {name} ({int(prune_ratio*100)}%)')
                        else:
                            if previous_pruned_local:
                                prune_layer(model, name, preserved_indexes_local, 1)
                                previous_pruned = False
                            if cnt in layers_to_prune:
                                if idx_in_bottleneck in [1,2] or cnt in [3,13,26,45]:
                                    idx = layers_to_prune.index(cnt)
                                    preserved_indexes_local = preserved_indexes_all[idx]
                                    prune_ratio = prune_layer(model, name, preserved_indexes_local, 0)
                                    if cnt in [3,13,26,45]: preserved_indexes_residual = preserved_indexes_local
                                else:
                                    prune_ratio = prune_layer(model, name, preserved_indexes_residual, 0)
                                    preserved_indexes_local = preserved_indexes_residual
                                print(f'[{cnt}] Target: {name} ({int(prune_ratio*100)}%)')
                                previous_pruned = True
                        idx_in_bottleneck +=1
                        cnt+=1
                    elif isinstance(module, nn.BatchNorm2d) and previous_pruned:
                        prune_layer(model, name, preserved_indexes_local)
                preserved_indexes = preserved_indexes_local
        if isinstance(block, nn.Linear) and previous_pruned: prune_layer(model, name_block, preserved_indexes, 1)

# --------------------------------------------------------------------------------

def prune_densenet40(model, preserved_indexes_all, layers_to_prune): 
    """
    Method to prune densenet40.
    """

    denseblock_size = 12
    last_select_index = [] # Conv indexes selected in the previous layer
    previous_pruned = False
    cnt=0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # prune
            if previous_pruned:
                prune_layer(model, name, last_select_index, 1)
            if cnt in layers_to_prune:
                idx = layers_to_prune.index(cnt)
                preserved_indexes = preserved_indexes_all[idx]
                prune_ratio = prune_layer(model, name, preserved_indexes, 0)
                print(f'[{cnt}] Target: {name} ({int(prune_ratio*100)}%)')
                previous_pruned = True
            else:
                preserved_indexes = list(range(int(module.weight.data.size(0))))
            # update last_select_index
            if cnt % (denseblock_size+1) == 0:
                last_select_index = preserved_indexes
                previous_pruned = cnt in layers_to_prune
            else:
                select_index = [x+(cnt+1)*denseblock_size-(cnt)//(denseblock_size+1)*denseblock_size for x in preserved_indexes]
                last_select_index += select_index
            cnt+=1
        elif isinstance(module, nn.BatchNorm2d):
            prune_layer(model, name, last_select_index)

# --------------------------------------------------------------------------------

def prune_googlenet(model, preserved_indexes_all, layers_to_prune):
    """
    Method to prune googlenet.
    """
    cnt = 0
    last_select_index = [] # Conv index selected in the previous layer
    previous_pruned = False

    for name_block, block in model.named_children():
        name_block = name_block.replace('module.', '') 
        if isinstance(block, nn.Sequential):
            for name_module, module in block.named_modules():
                name = f"{name_block}.{name_module}"
                if isinstance(module, nn.Conv2d):
                    if cnt in layers_to_prune:
                        idx = layers_to_prune.index(cnt)
                        preserved_indexes = preserved_indexes_all[idx]
                        prune_ratio = prune_layer(model, name, preserved_indexes, 0)
                        print(f'[{cnt}] Target: {name} ({int(prune_ratio*100)}%)')
                        previous_pruned = True
                        last_select_index = preserved_indexes
                    cnt+=1
                elif isinstance(module, nn.BatchNorm2d):
                    prune_layer(model, name, last_select_index)
        # === INCEPTION BLOCK (contains 4 parallel branches) ===
        if isinstance(block, Inception):
            initial_previous_pruned = previous_pruned      # same init for all branches
            previous_pruned = False
            initial_last_select_index = last_select_index  # same init for all branches
            last_select_index = []
            max_size = 0
            # ==== BRANCH ===
            for name_branch, branch in block.named_children(): # Branch in Inception block
                previous_pruned_local = initial_previous_pruned
                last_select_index_local = initial_last_select_index
                # ==== MODULES IN BRANCH ===
                for name_module, module in branch.named_modules():
                    name = f"{name_block}.{name_branch}.{name_module}"
                    if isinstance(module, nn.Conv2d):
                        cout = int(module.weight.data.size(0))
                        if previous_pruned_local:
                            prune_layer(model, name, last_select_index_local, 1)
                        if cnt in layers_to_prune:
                            idx = layers_to_prune.index(cnt)
                            preserved_indexes = preserved_indexes_all[idx]
                            prune_ratio = prune_layer(model, name, preserved_indexes, 0)
                            print(f'[{cnt}] Target: {name} ({int(prune_ratio*100)}%)')
                        else:
                            preserved_indexes = list(range(cout))
                        # update prev_filters_pruned
                        previous_pruned_local = cnt in layers_to_prune
                        # update last_select_index_local
                        last_select_index_local = preserved_indexes
                        cnt+=1
                    elif isinstance(module, nn.BatchNorm2d):
                        prune_layer(model, name, last_select_index_local)
                if previous_pruned_local:
                    # if we pruned the last conv layer of one of the branches
                    # then the output of the Inception block (= concatenatin of output of all the branches) is pruned
                    previous_pruned = True
                last_select_index += [x+max_size for x in preserved_indexes]
                max_size += cout
        if isinstance(block, nn.Linear) and previous_pruned: prune_layer(model, name_block, last_select_index, 1)
