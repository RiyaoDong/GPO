#cifar10 resnet32
#python baseline.py --dataset cifar10 --arch resnet --depth 32 --save_dir c10_r32_s-8_b1 --gpu-id 1

#cifar100 vgg19 90% sparsity
#python baseline.py --dataset cifar100 --arch vgg19_bn --depth 19 --s_init -30 --s_decay 0.00001 --b_init 0.0 --b_decay 0.0001 --save_dir STR_c100_v19_s-30_sd00001_b0_bd0001 --gpu-id 1
#python baseline.py --dataset cifar100 --arch vgg19_bn --depth 19 --s_init -30 --s_decay 0.00004 --b_init 0.0 --b_decay 0.0001 --save_dir STR_c100_v19_s-30_sd00004_b0_bd0001 --gpu-id 1
#python baseline.py --dataset cifar100 --arch vgg19_bn --depth 19 --s_init -30 --s_decay 0.00007 --b_init 0.0 --b_decay 0.0001 --save_dir STR_c100_v19_s-30_sd00007_b0_bd0001 --gpu-id 1

python baseline.py --dataset tinyimagenet --arch vgg19_bn --depth 19 --s_init -30 --s_decay 0.00001 --b_init 0.0 --b_decay 0.0001 --save_dir STR_tiny_v19_s-30_sd00001_b0_bd0001 --gpu-id 1
python baseline.py --dataset tinyimagenet --arch vgg19_bn --depth 19 --s_init -30 --s_decay 0.00004 --b_init 0.0 --b_decay 0.0001 --save_dir STR_tiny_v19_s-30_sd00004_b0_bd0001 --gpu-id 1
python baseline.py --dataset tinyimagenet --arch vgg19_bn --depth 19 --s_init -30 --s_decay 0.00007 --b_init 0.0 --b_decay 0.0001 --save_dir STR_tiny_v19_s-30_sd00007_b0_bd0001 --gpu-id 1

#cifar100 vgg19 90% sparsity
#python baseline.py --dataset tinyimagenet --arch vgg19_bn --depth 19 --s_init -30 --s_decay 0.00001 --b_init 1.0 --b_decay 0.0001 --save_dir Xtiny_v19_s-30_sd00001_b1_bd0001 --gpu-id 1
#python baseline.py --dataset tinyimagenet --arch vgg19_bn --depth 19 --s_init -30 --s_decay 0.00004 --b_init 1.0 --b_decay 0.0001 --save_dir tiny_v19_s-30_sd00004_b1_bd0001 --gpu-id 1
#python baseline.py --dataset tinyimagenet --arch vgg19_bn --depth 19 --s_init -30 --s_decay 0.00007 --b_init 1.0 --b_decay 0.0001 --save_dir tiny_v19_s-30_sd00007_b1_bd0001 --gpu-id 1
