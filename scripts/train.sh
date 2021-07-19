# Train a model using an multi-channel adaptor (or from scratch). 
# The model uses an ImageNet pre-trained VGG-D, ResNet18, or ResNet50. 
# Can be trained using CUB, Aircrafts, Cars, LEGUS, So2Sat, or So2Sat datasets.

CUDA_VISIBLE_DEVICES=1 python src/train.py \
                   	--batch-size 64 \
                        --epochs 30\
                   	--lr 1e-4 \
                   	--model none \
			--backbone vgg16 \
			--pretrained_adaptor 0 \
			--inflate 0 \
			--num_views 0 \
			--channels 3 \
			--dataset cub \
			--numdata 5994 \
			


# -model: adaptor type (linear, medium, none) none: from scratch, medium: multi-layer
# -dataset: (cub, cars, aircraft, legus, so2sat, eurosat)
# -numdata: number of training samples
#    Max. number of training samples:
#    cub: 200-5994 (3, 5, 15 channels)
#    cars: 196-8144 (3, 5, 15 channels)
#    aircrafts: 100-6667 (3, 5, 15 channels)
#    legus: 4-12376 (5 channels)
#    eurosat: 10-10000 (13 channels)
#    so2sat: 17-352366 (18 channels)
# -channels: number of channels (5, 15)
# -pretrained_adaptor: to use pre-trained adaptor (1, 0)

