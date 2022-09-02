## Demo

We have created a demo with EuroSAT dataset and multi-view linear adaptors for VGG16, ResNet18, and ResNet50. Please refer to `eurosat.py` folder for the sample dataloader and `mvcnn.py` for the multi-view linear adaptor. You
can change hyper-parameters, backbone architecture, and number of views in `demo.sh`:

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
bash demo.sh
```

