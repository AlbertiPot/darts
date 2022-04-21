# # search on c10
# CUDA_VISIBLE_DEVICES=5 nohup python cnn/train_search.py --data 'cnn/data/cifar/cifar10/' \
# --save 'c10_v2c10seed2' \
# --seed 2 \
# --unrolled \
# --learning_rate_min 0 &

# search on c100
# CUDA_VISIBLE_DEVICES=5 nohup python cnn/train_search.py --data 'cnn/data/cifar/cifar100/' \
# --unrolled \
# --save 'c100_v2c100seed2' \
# --seed 2 \
# --cifar100 \
# --learning_rate_min 0 &

# eval on c10
# CUDA_VISIBLE_DEVICES=0 nohup python cnn/train.py --data 'cnn/data/cifar/cifar10/' \
# --auxiliary \
# --cutout \
# --save 'c10_v2c10seed9999_trainseed0' \
# --arch v2c10seed9999 &

# eval on c100
CUDA_VISIBLE_DEVICES=4 nohup python cnn/train.py --data 'cnn/data/cifar/cifar100/' \
--auxiliary \
--cutout \
--save 'c100_v2c10seed2_trainseed0' \
--cifar100 \
--arch v2c10seed2 \
--weight_decay 5e-4 &