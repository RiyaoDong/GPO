#cifar10 resnet32
#python baseline.py --dataset cifar10 --arch resnet --depth 32 --save_dir c10_r32_s-8_b1 --gpu-id 1

#cifar100 resnet32 90% sparsity
#python baseline.py --dataset cifar100 --arch resnet --depth 32 --s_init -30 --s_decay 0.00002 --b_init 0.0 --b_decay 0.0001 --save_dir STR_c100_r32_s-30_sd00002_b0_bd0001 --gpu-id 0
#python baseline.py --dataset cifar100 --arch resnet --depth 32 --s_init -30 --s_decay 0.00005 --b_init 0.0 --b_decay 0.0001 --save_dir STR_c100_r32_s-30_sd00005_b0_bd0001 --gpu-id 0
#python baseline.py --dataset cifar100 --arch resnet --depth 32 --s_init -30 --s_decay 0.00008 --b_init 0.0 --b_decay 0.0001 --save_dir STR_c100_r32_s-30_sd00008_b0_bd0001 --gpu-id 0

python baseline.py --dataset tinyimagenet --arch resnet --depth 32 --s_init -30 --s_decay 0.00002 --b_init 0.0 --b_decay 0.0001 --save_dir STR_tiny_r32_s-30_sd00002_b0_bd0001 --gpu-id 0
python baseline.py --dataset tinyimagenet --arch resnet --depth 32 --s_init -30 --s_decay 0.00005 --b_init 0.0 --b_decay 0.0001 --save_dir STR_tiny_r32_s-30_sd00005_b0_bd0001 --gpu-id 0
python baseline.py --dataset tinyimagenet --arch resnet --depth 32 --s_init -30 --s_decay 0.00008 --b_init 0.0 --b_decay 0.0001 --save_dir STR_tiny_r32_s-30_sd00008_b0_bd0001 --gpu-id 0

#cifar100 resnet32 95% sparsity
#python baseline.py --dataset tinyimagenet --arch resnet --depth 32 --s_init -30 --s_decay 0.00002 --b_init 1.0 --b_decay 0.0001 --save_dir tiny_r32_s-30_sd00002_b1_bd0001 --gpu-id 0
#python baseline.py --dataset tinyimagenet --arch resnet --depth 32 --s_init -30 --s_decay 0.00005 --b_init 1.0 --b_decay 0.0001 --save_dir tiny_r32_s-30_sd00005_b1_bd0001 --gpu-id 0
#python baseline.py --dataset tinyimagenet --arch resnet --depth 32 --s_init -30 --s_decay 0.00008 --b_init 1.0 --b_decay 0.0001 --save_dir tiny_r32_s-30_sd00008_b1_bd0001 --gpu-id 0
