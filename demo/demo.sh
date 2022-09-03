# Demo to train a model using an multi-view linear adaptor. 
# The model uses an ImageNet pre-trained VGG-D, ResNet18, or ResNet50. 
# This demo is prepared for EuroSAT dataset (http://madm.dfki.de/downloads).

CUDA_VISIBLE_DEVICES=0 python train.py \
                        --batch-size 32 \
                        --epochs 1 \
                        --lr 1e-4 \
                        --backbone vgg16 \
                        --num_views 5 \


