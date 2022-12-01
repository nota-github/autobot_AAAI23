# Automatic Neural Network Pruning that Efficiently Preserves the Model Accuracy

## How to use this code :gear: 

Minimal code (use default parameters):
```
python3 main.py
```

To prune the network with the default MFLOPs target, use the following command:
```
python3 main.py --mode prune --arch architecture_name --dataset dataset_name --output_dir result/pruned/
```

To finetune a pruned network, use the following command:
```
python3 main.py --mode finetune --loaded_model_path path/to/pruned.pt --arch architecture_name --dataset dataset_name --output_dir result/finetuned/
```

The datasets are assumed to be saved in *./data/name_of_the_dataset/*.

### Parameters

`--loaded_model_path`
- the input model that we want to compress is loaded from this path.
- if it is a directory, then by default the loaded file will be *`--arch`.pt*.

`--output_dir`
- the output compressed model, as well as the logs, will be saved at this path.

`--arch`
- it is the architecture on which your model is based
- currently accepted values are: vgg_16_bn, resnet_56, resnet_110, densenet_40, googlenet, resnet_50

`--Mflops_target`
- it is the targetted number of MFLOPs of the pruned model (the output model will be as close as possible to this target)

For the other parameters, feel free to check the *[parser.py](utils/parser.py)* file.

## Results :medal_sports:

### VGG

```
python3 main.py --output_dir result/pruned/vgg16/ --mode prune --arch vgg_16_bn
python3 main.py --output_dir result/finetuned/vgg16/ --mode finetune --arch vgg_16_bn --loaded_model_path result/pruned/vgg16/pruned.pt
```

Initial accuracy: 93.96\
Initial MFLOPs: 314.294\
Dataset: CIFAR10

| Acc. before finetuning |    Acc. after finetuning    |    MFLOPs    |
|:----------------------:|:---------------------------:|:------------:|
|          88.29         |            94.19            |    145.61    |
|          82.73         |            94.01            |    108.71    |
|          71.24         |            93.62            |     72.60    |

### ResNet56

```
python3 main.py --output_dir result/pruned/resnet56/ --mode prune --arch resnet_56
python3 main.py --output_dir result/finetuned/resnet56/ --mode finetune --arch resnet_56 --loaded_model_path result/pruned/resnet56/pruned.pt
```

Initial accuracy: 93.26\
Initial MFLOPs: 126.55\
Dataset: CIFAR10

| Acc. before finetuning |    Acc. after finetuning    |    MFLOPs    |
|:----------------------:|:---------------------------:|:------------:|
|          85.58         |            93.76            |     55.82    |

### ResNet110

```
python3 main.py --output_dir result/pruned/resnet110/ --mode prune --arch resnet_110
python3 main.py --output_dir result/finetuned/resnet110/ --mode finetune --arch resnet_110 --loaded_model_path result/pruned/resnet110/pruned.pt
```

Initial accuracy: 93.5\
Initial MFLOPs: 254.99\
Dataset: CIFAR10

| Acc. before finetuning |    Acc. after finetuning    |    MFLOPs    |
|:----------------------:|:---------------------------:|:------------:|
|          84.37         |            94.15            |     85.28    |

### GoogleNet

```
python3 main.py --output_dir result/pruned/googlenet/ --mode prune --arch googlenet
python3 main.py --output_dir result/finetuned/googlenet/ --mode finetune --arch googlenet --loaded_model_path result/pruned/googlenet/pruned.pt
```

Initial accuracy: 95.05\
Initial MFLOPs: 1529.42\
Dataset: CIFAR10

| Acc. before finetuning |    Acc. after finetuning    |    MFLOPs    |
|:----------------------:|:---------------------------:|:------------:|
|          90.18         |            95.23            |      450     |

### DenseNet40

```
python3 main.py --output_dir result/pruned/densenet/ --mode prune --arch densenet_40
python3 main.py --output_dir result/finetuned/densenet/ --mode finetune --arch densenet_40 --loaded_model_path result/pruned/densenet/pruned.pt
```

Initial accuracy: 94.81\
Initial MFLOPs: 287.71\
Dataset: CIFAR10

| Acc. before finetuning |    Acc. after finetuning    |    MFLOPs    |
|:----------------------:|:---------------------------:|:------------:|
|          83.20         |            94.41            |    128.25    |
|          87.85         |            94.67            |    167.64    |

### ResNet50

```
python3 main.py --output_dir result/pruned/resnet50/ --mode prune --batch_size 64 --nb_batches 3000 --arch resnet_50 --beta 10.0 --gamma 0.3 --lr 0.3 --dataset imagenet
python3 main.py --output_dir result/finetuned/resnet50/ --mode finetune --batch_size 512 --wd 0.0001 --lr_finetuning 0.006 --momentum 0.99 --arch resnet_50 --loaded_model_path result/pruned/resnet50/pruned.pt --dataset imagenet
```


Initial accuracy: 76.13\
Initial MFLOPs: 4111.51\
Dataset: ImageNet

| Acc. before finetuning |    Acc. after finetuning    |    MFLOPs    |
|:----------------------:|:---------------------------:|:------------:|
|          14.71         |            74.68            |     1149     |
|          47.51         |            76.63            |     1974     |


## Requirements :wrench:
* pytorch
* thop