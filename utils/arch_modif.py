import torch.nn as nn

def get_new_layer(orig_layer, new_cin=None, new_cout=None):
    """ Generate a new layer, with the properties of the input layer.
    If the new number of input or output channel changed, return a new layer with the updated dimensions.
    Else, return the input layer.
    Remark: if the dim is modified, the original data is not preserved!
    """
    if isinstance(orig_layer, nn.ReLU):
        return orig_layer

    if len(orig_layer.weight.data.size()) > 1:
        orig_cin = orig_layer.weight.data.size(1)
        orig_cout = orig_layer.weight.data.size(0)
    else: orig_cin = orig_cout = orig_layer.weight.data.size(0)
    if new_cin is None: new_cin = orig_cin
    if new_cout is None: new_cout = orig_cout
    if (orig_cin == new_cin and orig_cout == new_cout):
        return orig_layer
        
    if isinstance(orig_layer, nn.Conv2d):
        new_layer = nn.Conv2d(
                            in_channels = new_cin, 
                            out_channels = new_cout,
                            kernel_size =  orig_layer.kernel_size,
                            stride =       orig_layer.stride,
                            padding =      orig_layer.padding,
                            dilation =     orig_layer.dilation,
                            bias =         orig_layer.bias is not None
                        )
    if isinstance(orig_layer, nn.Linear):
        new_layer = nn.Linear(
                            in_features =  new_cin, 
                            out_features = new_cout,
                            bias =         orig_layer.bias is not None
                        )
    if isinstance(orig_layer, nn.BatchNorm2d):
        new_c = new_cin if orig_cin != new_cin else new_cout
        new_layer = nn.BatchNorm2d(
                            num_features = new_c, 
                            eps =          orig_layer.eps, 
                            momentum =     orig_layer.momentum, 
                            affine =       orig_layer.affine, 
                            track_running_stats = orig_layer.track_running_stats, 
        )
    new_layer = new_layer.to(next(orig_layer.parameters()).device)
    del orig_layer
    return new_layer

def replace_layer(model: nn.Module, target: nn.Module, replacement: nn.Module):
    """
    Replace a given module within a parent module with some third module
    Useful for injecting new layers in an existing model.
    """
    def replace_in(model: nn.Module, target: nn.Module, replacement: nn.Module):
        # print("searching ", model.__class__.__name__)
        for name, submodule in model.named_children():
            # print("is it member?", name, submodule == target)
            if submodule == target:
                # we found it!
                if isinstance(model, nn.ModuleList):
                    # replace in module list
                    model[name] = replacement

                elif isinstance(model, nn.Sequential):
                    # replace in sequential layer
                    if name.isdigit():
                        model[int(name)] = replacement
                    else:
                        set_module_in_model(model, name, replacement)
                else:
                    # replace as member
                    model.__setattr__(name, replacement)

                # print("Replaced " + target.__class__.__name__ + " with "+replacement.__class__.__name__+" in " + model.__class__.__name__)
                return True

            elif len(list(submodule.named_children())) > 0:
                # print("Browsing {} children...".format(len(list(submodule.named_children()))))
                if replace_in(submodule, target, replacement):
                    return True
        return False

    if not replace_in(model, target, replacement):
        raise RuntimeError("Cannot substitute layer: Layer of type " + target.__class__.__name__ + " is not a child of given parent of type " + model.__class__.__name__)


def set_module_in_model(model, name_prev_module, new_module):
    modules_names = name_prev_module.split(".")
    m = model
    for m_name in modules_names[:-1]:
        if m_name.isdigit():
            m = m[int(m_name)]
        else:
            m = getattr(m, m_name)
    m_name = modules_names[-1]
    if m_name.isdigit(): m[int(m_name)] = new_module
    else: setattr(m, m_name, new_module)
    return m

def get_module_in_model(model, name_module):
    modules_names = name_module.split(".")
    m = model
    for m_name in modules_names:
        if m_name.isdigit():
            m = m[int(m_name)]
        else:
            m = getattr(m, m_name)
    return m

def prune_layer(model: nn.Module, name_target: str, preserved_indexes: list, dim: int = 0):
    """
        Prune the given module (name_target) in the given model, in order to only preserve the filter at the given indexes.
        Args:
            - model: the model to prune
            - name_target: the name of the module to prune
            - preserved_indexes: the indexes of the filters to keep
            - dim: 0 for output channel, 1 for input channel (dimension in the weight.data tensor)
        Return: the pruning ratio
    """
    c_nb = len(preserved_indexes)
    orig_layer = get_module_in_model(model, name_target)

    if len(orig_layer.weight.data.size()) > 1:
        orig_cin = orig_layer.weight.data.size(1)
        orig_cout = orig_layer.weight.data.size(0)
    else: orig_cin = orig_cout = orig_layer.weight.data.size(0)

    if isinstance(orig_layer, (nn.Conv2d, nn.Linear)):
        original_weight_data = orig_layer.weight.data.detach().clone()
        if orig_layer.bias is not None: original_bias_data = orig_layer.bias.data.detach().clone()

        if dim==0: # prune output channel
            new_layer = get_new_layer(orig_layer, new_cin=None, new_cout=c_nb)
            new_layer.weight.data = original_weight_data[preserved_indexes]
            if new_layer.bias is not None: new_layer.bias.data = original_bias_data[preserved_indexes]
        else: # prune input channel
            new_layer = get_new_layer(orig_layer, new_cin=c_nb, new_cout=None)
            new_layer.weight.data = original_weight_data[:, preserved_indexes]
            if new_layer.bias is not None: new_layer.bias.data = original_bias_data
    elif isinstance(orig_layer, nn.BatchNorm2d):
        extensions = ['weight', 'bias', 'running_var', 'running_mean']
        saved_data = []
        for ext in extensions:
            if hasattr(orig_layer, ext):
                saved_data.append(eval('orig_layer.' + ext).data.detach().clone())
        new_layer = get_new_layer(orig_layer, new_cout=c_nb)
        i = 0
        for ext in extensions:
            if hasattr(new_layer, ext):
                eval('new_layer.' + ext).data = saved_data[i][preserved_indexes]
                i+=1
    set_module_in_model(model, name_target, new_layer)

    orig = (orig_cout if dim==0 else orig_cin)
    return (orig-len(preserved_indexes))/orig