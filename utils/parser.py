import os
import argparse
import sys

def get_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./result/test',
        help='Path to the directory where the outputs will be saved (checkpoints, logs, etc...)')

    parser.add_argument(
        '--loaded_model_path',
        type=str,
        default='./checkpoints/classification/',
        help='Path to the pretrain model.\n \
            If --loaded_model_path is a directory, then by default the loaded file will be arch_name.pt')

    parser.add_argument(
        '--resume',
        default=False,
        action='store_true',
        help='to resume a finetuning')

    parser.add_argument(
        '--test_only',
        default=False,
        action='store_true',
        help='If True, evalute only')

    parser.add_argument(
        '--mode',
        type=str,
        default='prune',
        choices=('prune', 'finetune'),
        help='Choose prune to prune the model and finetune to finetune the model')

    # -------------------------- bottleneck training params

    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size.')

    parser.add_argument(
        '--nb_batches',
        type=int,
        default=200,
        help='number of batches')

    parser.add_argument(
        '--Mflops_target',
        type=float,
        default= None,
        help='targetted M of FLOPS of the pruned model')

    # --------------------------  bottleneck training hyperparams

    parser.add_argument(
        '--lr',
        default=0.6,
        type=float,
        help='initial learning rate')

    parser.add_argument(
        '--momentum',
        default=0.9,
        type=float,
        help='momentum')

    parser.add_argument(
        '--beta',
        default=6,
        type=float,
        help='beta (weight of the FLOPs loss in the computation of the final loss)')

    parser.add_argument(
        '--gamma',
        default=0.4,
        type=float,
        help='gamma (weight of the boolean loss in the computation of the final loss)')

    # -------------------------- params hardware

    parser.add_argument(
        '--gpu',
        type=str,
        default='0',
        help='Select gpu to use')

    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='num of workers for dataloader'
    )

    # -------------------------- 

    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        # default='imagenet',
        choices=('cifar10', 'imagenet'),
        help='dataset')

    parser.add_argument(
        '--arch',
        type=str,
        default='resnet_56',
        choices=('vgg_16_bn', 'resnet_56', 'resnet_110', 'densenet_40', 'googlenet', 'resnet_50'),
        help='The architecture of the model')

    parser.add_argument(
        '--save_plot',
        default=False,
        action='store_true',
        help='whether save accuracy plots or not')

    parser.add_argument(
        '--seed', 
        type=int, 
        default=1, 
        metavar='S',
        help='random seed (default: 1)')

    # -------------------------- finetuning

    parser.add_argument(
        '--lr_finetuning',
        default=0.02,
        type=float,
        help='initial learning rate for finetuning')

    parser.add_argument(
        '--epoch_finetuning',
        default=200,
        type=int,
        help='nb epochs for finetuning')

    parser.add_argument(
        '--wd',
        default=0.002,
        type=float,
        help='weight decay for finetuning')


    # -------------------------- 

    args = parser.parse_args(args)

    if not os.path.isfile(args.loaded_model_path):
        args.loaded_model_path = os.path.join(args.loaded_model_path, args.dataset)
        filename = args.arch + '.pt'
        args.loaded_model_path = os.path.join(args.loaded_model_path, filename)
        
    # Data Acquisition
    args.data_dir = {
        "cifar10": './data/cifar10/',  # CIFAR-10
        "imagenet": './data/imagenet/',  # ImageNet
    }[args.dataset]
    return args


