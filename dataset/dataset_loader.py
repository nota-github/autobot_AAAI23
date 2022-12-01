
from dataset.cifar10 import load_data as load_data_cifar10
from dataset.imagenet import load_data as load_data_imagenet

def load_data(args):
    return eval(f'load_data_{args.dataset}')(args)