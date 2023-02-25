# # search on c10
# CUDA_VISIBLE_DEVICES=4 nohup python cnn/train_search.py --data 'cnn/data/cifar/cifar10/' \
# --save 'c10_v1c10seed9999' \
# --seed 9999 \
# --learning_rate_min 0 &
# --unrolled \

# search on c100
# CUDA_VISIBLE_DEVICES=2 nohup python cnn/train_search.py --data 'cnn/data/cifar/cifar100/' \
# --save 'c100_v1c100seed2' \
# --seed 2 \
# --cifar100 \
# --learning_rate_min 0 &
#--unrolled \

# eval on c10
# CUDA_VISIBLE_DEVICES=0 nohup python cnn/train.py --data 'cnn/data/cifar/cifar10/' \
# --auxiliary \
# --cutout \
# --save 'c10_v2c10seed9999_trainseed0' \
# --arch v2c10seed9999 &

# eval on c100
# CUDA_VISIBLE_DEVICES=4 nohup python cnn/train.py --data 'cnn/data/cifar/cifar100/' \
# --auxiliary \
# --cutout \
# --save 'c100_v1c10seed9999_trainseed0' \
# --cifar100 \
# --arch v1c10seed9999 \
# --weight_decay 5e-4 &

#debug
# CUDA_VISIBLE_DEVICES=4 python cnn/train.py --data 'cnn/data/cifar/cifar100/' \
# --auxiliary \
# --cutout \
# --save 'debug_cutout' \
# --cifar100 \
# --arch v2c10seed2

# CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/rookie/bin/python /data/gbc/Workspace/darts/cnn/train_search.py --data '/data/gbc/Datasets/cifar/cifar10' \
# --save 'timecost_c10_v1c10seed9999' \
# --seed 9999 \
# --learning_rate_min 0 \
# --unrolled

# CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/rookie/bin/python /data/gbc/Workspace/darts/cnn/train_search.py --data '/data/gbc/Datasets/cifar/cifar100' \
# --save 'timecost_c100_v2c100seed9999' \
# --seed 9999 \
# --learning_rate_min 0 \
# --cifar100 --unrolled

# timecost test
# python train_search.py \
# --data '/data/usr/gbc/dataset/cifar/cifar10' \
# --save 'timecost_c10_v1c10seed9999' \
# --seed 2 \
# --learning_rate_min 0

# visual on c10
CUDA_VISIBLE_DEVICES=1 python cnn/train_search.py \
--data '/data/usr/gbc/dataset/cifar/cifar10' \
--save 'visual_c10_v1c10seed2' \
--seed 2 \
--learning_rate_min 0 \
--visual