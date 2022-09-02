# Hyperspectral domain adaptors

PyTorch code for training and evaluation of the hyperspectral domain adaptors for the six datasets mentioned below.

## Table of contents
* [Getting started](#getting-started)
* [Datasets](#datasets)
* [Demo](#demo)
* [Oracle fine-tuning](#oracle-fine-tuning-with-imagenet-pre-trained-vgg-d)
* [Pre-training adaptors](#pre-training-adaptors)
* [Fine-tuning models for hyperspectral datasets](#fine-tuning-models-for-hyperspectral-datasets)

## Getting started

In this section we show how to setup the repository, install virtual environments (Virtualenv or Anaconda), and install requirements.

<details>
<summary>Click to expand</summary>

1. **Clone the repository:** To download this repository run:
```
$ git clone https://github.com/gperezs/Hyperspectral_domain_adaptors.git
$ cd Multispectral_domain_adaptation
```

In the following sections we show two ways to setup StarcNet. Use the one that suits you best:
* [Using virtualenv](#using-virtualenv)
* [Using Anaconda](#using-anaconda)

### Using virtualenv

2. **Install virtualenv:** To install virtualenv run after installing pip:

```
$ sudo pip3 install virtualenv
```

3. **Virtualenv  environment:** To set up and activate the virtual environment,
run:
```
$ virtualenv -p /usr/bin/python3 venv3
$ source venv3/bin/activate
```

To install requirements, run:
```
$ pip install -r requirements.txt
```

4. **PyTorch:** To install pytorch run:
```
$ pip install torch torchvision
```

-------
### Using Anaconda

2. **Install Anaconda:** We recommend using the free [Anaconda Python
distribution](https://www.anaconda.com/download/), which provides an
easy way for you to handle package dependencies. Please be sure to
download the Python 3 version.

3. **Anaconda virtual environment:** To set up and activate the virtual environment,
run:
```
$ conda create -n <env name> python=3.*
$ source activate <env name>
```

To install requirements, run:
```
$ conda install --yes --file requirements.txt
```

4. **PyTorch:** To install pytorch follow the instructions [here](https://pytorch.org/).
</details>

## Datasets

To get started download datasets listed below. Add each dataset directory where you download the data in `config.py`.

### Synthetic datasets
* Caltech-UCSD Birds (200 classes): [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).
* FGVC Aircrafts (100 classes): [FGVC aircraft dataset](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/).
* Stanford Cars (196 classes): [Stanford cars dataset](http://ai.stanford.edu/~jkrause/cars/car_dataset.html).

### Realistic datasets
* Legacy ExtraGalactic UV Survey (4 classes): [LEGUS dataset](https://archive.stsci.edu/prepds/legus/).
* So2Sat LCZ42 (17 classes): [So2Sat LCZ42 dataset](https://mediatum.ub.tum.de/1483140). 
* EuroSAT (10 classes): [EuroSAT dataset (13 bands)](http://madm.dfki.de/downloads).

## Demo

We have created a demo with EuroSAT dataset and multi-view linear adaptors for VGG16, ResNet18, and ResNet50. Please refer to `demo/eurosat.py` folder for the sample dataloader and `demo/mvcnn.py` for the multi-view linear adaptor. You can change hyper-parameters, backbone architecture, and number of views in `demo/demo.sh`:

```
CUDA_VISIBLE_DEVICES=0 python train.py \
                        --batch-size 64 \
                        --epochs 30\
                        --lr 1e-4 \
                        --backbone vgg16 \
                        --num_views 5 \
```

To run demo:
```
cd demo
bash demo.sh
```

## Oracle fine-tuning with ImageNet pre-trained VGG-D

To perform oracle fine-tune on VGG-D with one of the datasets (CUB, Cars, or Aircraft) run:
```
bash scripts/standard_finetune.sh
```
Inside the bash file you can set the options for datasets (`dataset: cub, cars, aircrafts`), backbone model (`model: vgg16 or resnet18`), and number of training samples (`numdata`).
```
CUDA_VISIBLE_DEVICES=0 python src/standard_finetune.py \
                        --batch-size 64 \
                        --epochs 30\
                        --lr 1e-4 \
                        --model vgg16 \
			--dataset cub \
                        --numdata 5994 \
```

Here we show results fine-tuning an ImageNet pre-trained VGG-D, ResNet18, and ResNet50 for CUB, Cars, and Aircraft datasets.

| Datasets    |  VGG-D   |  ResNet18   |  ResNet50   |
| :---        | :----:  |  :----: |  :----: |
| CUB-200-2011       | 72.0%  |  70.5% |  77.0% |
| Stanford Cars      | 78.2%  |  80.6% |  86.7% | 
| FGVC Aircraft      | 81.1%  |  76.9% |  84.0% | 


## Pre-training adaptors

The multi-layer adaptor is pre-trained unsupervisely using an autoencoder scheme. 

To pre-train the multi-layer adaptor run:
```
bash scripts/pretrain_adaptor.sh
```
Inside the bash file you can set the options for datasets (`dataset: cub, cars, aircrafts, legus, sunrgbd, eurosat`)
, number of training samples (`numdata`), and number of channels (`channels: 5 or 15`). 
```
CUDA_VISIBLE_DEVICES=0 python src/pretrain_adaptor.py \
                        --batch-size 64 \
                        --epochs 50\
                        --lr 1e-3 \
                        --dataset cub \
                        --numdata 5994 \
                        --channels 5 \
```


## Fine-tuning models for hyperspectral datasets (Adaptors + VGG-D/ResNet18/ResNet50)

To fine-tune a VGG-D, ResNet18, or ResNet50 pretrained model using hyperspectral datasets (synthetic or realistic) run:
```
bash scripts/train.sh
```
Inside the bash file you can set the options for datasets (`dataset: cub, cars, aircrafts, legus, so2sat, eurosat`),
backbone model (`backbone: vgg16 or resnet18`), number of training samples (`numdata`), number of channels (`channels: 
5 or 15`), adaptor type (`model: linear or medium, none`), using pre-trained adaptors or randomly initialized, 
using inflated filters, and number of views for multi-view scheme (`num_views`). `medium` refers to multi-layer adaptor.

* To do standard fine-tuning with 3-channel images set `model = none` and number of `channels = 3`.
* To train network from scratch set `model = none`, number of `channels != 3`, and `inflate = 0`.
* To use inflated filters in the first conv layer set `model = none`, number of `channels != 3`, and `inflate = 1`.
* To use learnable adaptors (linear or multi-layer) trained from scratch set `model != none` and `pretrained_adaptor = 0`.
* To use pre-trained learnable multi-layer adaptor set `model = medium` and `pretrained_adaptor = 1` (adaptor had to be pre-trained before).
* To train using a multi-view scheme with random subset sampling set `model = none` and `num_views > 0`.
* To train using a multi-view scheme with linear adaptor set `model = linear` and `num_views > 0`.

```
CUDA_VISIBLE_DEVICES=0 python src/train.py \
                        --batch-size 32 \
                        --epochs 15\
                        --lr 1e-4 \
                        --model medium \
			--backbone vgg16 \
                        --pretrained_adaptor 1 \
                        --inflate 0 \
			--num_views 0 \
			--channels 5 \
                        --dataset legus \
                        --numdata 12376 \
```





