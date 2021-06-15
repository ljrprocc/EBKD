# sample scripts for training vanilla teacher models

# python train_teacher.py --model wrn_40_2

# python train_teacher.py --model resnet56

# python train_teacher.py --model resnet110

# CUDA_VISIBLE_DEVICES=4 python train_teacher.py --model resnet32x4 --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --resume --learning_rate 0.001 --lr_decay_epochs 30,60,100 --trial 3 --mode G --save_freq 2

# python train_teacher.py --model vgg13

CUDA_VISIBLE_DEVICES=6 python train_ebm.py --model_s resnet8x4 --trial 1_var --capcitiy 40000 --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --g_steps 20 --save_freq 50 --epochs 500 --lr_decay_epochs 100,250 --plot_cond --learning_rate 0.0001 --batch_size 128  --energy mcmc --warmup_iters 1000 --lmda_p_x_y 1. --lmda_p_x 0 --step_size 5 --act leaky --lmda_kl 1
