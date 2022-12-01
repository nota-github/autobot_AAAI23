import os
import time, datetime
import utils.common as utils
from utils.parser import get_args

import torch
import torch.backends.cudnn as cudnn
import numpy as np

from modules.finetuner import Finetuner

args = get_args()

# Data Acquisition
args.print_freq = { 
    "cifar10": (256*50)//args.batch_size,  # CIFAR-10
    "imagenet": (256*500)//args.batch_size  # ImageNet
}[args.dataset]

args.num_classes = {
    "cifar10": 10,     # CIFAR-10
    "imagenet": 1000   # ImageNet
}[args.dataset]

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.device_ids = list(map(int, args.gpu.split(',')))

torch.manual_seed(args.seed)  ##for cpu
np.random.seed(0)
if args.gpu:
    torch.cuda.manual_seed(args.seed)  ##for gpu

if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

utils.record_config(args)
now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
logger = utils.get_logger(os.path.join(args.output_dir, 'logger'+now+'.log'))

#use for loading pretrain model
if len(args.gpu)>1:
    args.name_base='module.'
else:
    args.name_base=''

def main():
    start_t = time.time()

    cudnn.benchmark = True
    cudnn.enabled=True
    logger.info("args = %s", args)

    tuner = Finetuner(args, logger) #import the network in a predefined class

    if not args.test_only:
        if args.resume:
            tuner.finetune()
            tuner.save_onnx()
        else:
            if args.mode == "prune":
                tuner.prune_model()
                tuner.show_results()
            elif args.mode == "finetune":
                tuner.finetune()
                tuner.save_onnx()
                tuner.show_results()
    else:
        tuner.show_results()
            
    end_t = time.time()
    
    logger.info('-'*40)
    logger.info(f'Total time: {(end_t - start_t):.2f}s')


if __name__ == '__main__':
    main()
