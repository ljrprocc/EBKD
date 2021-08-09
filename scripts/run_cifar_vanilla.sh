# sample scripts for training vanilla teacher models

# python train_teacher.py --model wrn_40_2

CUDA_VISIBLE_DEVICES=3 python train_teacher.py --model vgg13 --batch_size 512 --dataset svhn --print_freq 20 --learning_rate_ebm 0.01 --epochs 100 --lr_decay_epochs_ebm 50,75

CUDA_VISIBLE_DEVICES=3 python train_teacher.py --model resnet32x4 --batch_size 512 --dataset svhn --print_freq 20 --learning_rate_ebm 0.01 --epochs 100 --lr_decay_epochs_ebm 50,75

CUDA_VISIBLE_DEVICES=3 python train_teacher.py --model wrn_40_2 --batch_size 512 --dataset svhn --print_freq 20 --learning_rate_ebm 0.01 --epochs 100 --lr_decay_epochs_ebm 50,75

CUDA_VISIBLE_DEVICES=3 python train_teacher.py --model vgg16 --batch_size 512 --dataset svhn --print_freq 20 --learning_rate_ebm 0.01 --epochs 100 --lr_decay_epochs_ebm 50,75

CUDA_VISIBLE_DEVICES=3 python train_teacher.py --model vgg19 --batch_size 512 --dataset svhn --print_freq 20 --learning_rate_ebm 0.01 --epochs 100 --lr_decay_epochs_ebm 50,75

CUDA_VISIBLE_DEVICES=3 python train_teacher.py --model resnet110 --batch_size 512 --dataset svhn --print_freq 20 --learning_rate_ebm 0.01 --epochs 100 --lr_decay_epochs_ebm 50,75

CUDA_VISIBLE_DEVICES=3 python train_teacher.py --model resnet56 --batch_size 512 --dataset svhn --print_freq 20 --learning_rate_ebm 0.01 --epochs 100 --lr_decay_epochs_ebm 50,75

# python train_teacher.py --model resnet110

# CUDA_VISIBLE_DEVICES=4 python train_teacher.py --model resnet32x4 --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --resume --learning_rate 0.001 --lr_decay_epochs 30,60,100 --trial 3 --mode G --save_freq 2

# python train_teacher.py --model vgg13

# CUDA_VISIBLE_DEVICES=1 python train_teacher.py --model resnet110 --capcitiy 20000
