CUDA_VISIBLE_DEVICES=0 python src/standard_finetune.py \
                   	--batch-size 64 \
                   	--epochs 30 \
			--model vgg16 \
                   	--lr 1e-4 \
			--dataset cub \
			--numdata 5994 \

# -dataset: (cub, cars, aircraft)
# -numdata: number of training samples
#    Max. number of training samples:
#    cub: 200-5994
#    cars: 196-8144
#    aircrafts: 100-6667
