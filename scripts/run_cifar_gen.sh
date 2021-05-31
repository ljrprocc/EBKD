# sample scripts for training vanilla teacher models

# python train_teacher.py --model wrn_40_2

# python train_teacher.py --model resnet56

# python train_teacher.py --model resnet110

# CUDA_VISIBLE_DEVICES=4 python train_teacher.py --model resnet32x4 --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --resume --learning_rate 0.001 --lr_decay_epochs 30,60,100 --trial 3 --mode G --save_freq 2

# python train_teacher.py --model vgg13

CUDA_VISIBLE_DEVICES=1 python train_ebm.py --model resnet32x4 --capcitiy 20000 --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --steps 20 --save_freq 5 --epochs 70 --lr_decay_epochs 30,40 --plot_uncond --learning_rate 0.0001
