CUDA_VISIBLE_DEVICES=2 python src/pretrain_adaptor_MI.py \
                   	--batch-size 64 \
                   	--epochs 50\
                   	--lr 1e-3 \
			--dataset cub \
			--numdata 5994 \
			--channels 5 \

# -dataset: (cub, cars, aircraft)
# -numdata: number of training samples
#    Max. number of training samples:
#    cub: 5994
#    cars: 8144
#    aircrafts: 6667
# -channels: number of channels (5, 15)
