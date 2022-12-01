#reference: https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html#running-the-model-on-an-image-using-onnx-runtime

import torch
import torch.onnx
from utils.parser import get_args
from torch.nn import Parameter

from models.vgg import vgg_16_bn
from models.resnet_cifar10 import resnet_56, resnet_110
from models.googlenet import googlenet, Inception
from models.densenet import densenet_40
from models.mobilenetv1 import mobilenet_v1
from models.mobilenetv2 import mobilenet_v2
from models.minimalistnet import minimalistnet
import os
from utils.arch_modif import prune_layer
from utils.model_loading import load

args = get_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = 1
path = '/ssd8/thibault/code/IBA_pruning/result/final/vgg/lrfntnng0v03000_wd0v00200_mflpstrgt73v70000'
path = path + '/seed_01/'
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(path + 'model_best.pt', map_location=args.device)

model = eval(args.arch)()

model = load(model, checkpoint["state_dict"], args.arch)

# Input to the model
model.eval()
x = torch.randn(batch_size, 3, 32, 32, requires_grad=True)
torch_out = model(x)


# Export the model
torch.onnx.export(model,                     # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "vgg_16.onnx",             # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})


# #When open onnx file
# import onnx
#
# onnx_model = onnx.load("vgg_16.onnx")
# onnx.checker.check_model(onnx_model)
#
# import onnxruntime
# import numpy as np
#
# ort_session = onnxruntime.InferenceSession("vgg_16.onnx")
#
# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
#
# # compute ONNX Runtime output prediction
# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
# ort_outs = ort_session.run(None, ort_inputs)
#
# # compare ONNX Runtime and PyTorch results
# np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
#
# print("Exported model has been tested with ONNXRuntime, and the result looks good!")