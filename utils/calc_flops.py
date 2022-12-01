import torch
import torch.nn as nn

def get_flops(module_type, info, lamb_in=None, lamb_out=None, train=True):
    """
    Remark: these functions are adapted to match the thop library
    """
    cin = info['cin'] if lamb_in is None else sum([torch.sum(l) for l in lamb_in])
    cout = info['cout'] if lamb_out is None else sum([torch.sum(l) for l in lamb_out])

    if not train:
        if cin is not None: cin = float(cin)
        if cout is not None: cout = float(cout)
    # print()
    # print(cin)
    # print(cout)
    # print()

    assert(cin<=info['cin'] and cout<=info['cout'])
    hw = info['h'] * info['w']
    if module_type is nn.Conv2d:
        return get_flops_conv2d(info, cin, cout, hw)
    elif module_type is nn.Linear:
        return get_flops_linear(info, cin, cout, hw)
    elif module_type is nn.BatchNorm2d:
        return get_flops_batchnorm2d(info, cin, cout, hw)
    elif module_type is nn.ReLU:
        return get_flops_ReLU(info, cin, cout, hw)
    elif module_type is nn.AdaptiveAvgPool2d:
        return get_flops_AdaptiveAvgPool2d(info, cin, cout, hw)
    elif module_type is nn.AvgPool2d:
        return get_flops_AvgPool2d(info, cin, cout, hw)
    else: return 0

def get_flops_conv2d(info, cin, cout, hw):
    total_flops = 0
    kk = info['k']**2 
    # bias = Cout x H x W if model has bias, else 0
    bias_flops = 0 #cout*hw if info['has_bias'] else 0
    # Cout x Cin x H x W x Kw x Kh
    return cout * cin * hw * kk + bias_flops

def get_flops_linear(info, cin, cout, hw):
    #bias_flops = cout if info['has_bias'] else 0
    return cin * cout #* hw + bias_flops

def get_flops_batchnorm2d(info, cin, cout, hw):
    return hw * cin * 2

def get_flops_ReLU(info, cin, cout, hw):
    return 0 #hw * cin

def get_flops_AdaptiveAvgPool2d(info, cin, cout, hw):
    kk = info['k']**2
    return (kk+1) * hw * cout

def get_flops_AvgPool2d(info, cin, cout, hw):
    return hw * cout