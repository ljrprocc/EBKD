# sample scripts for training vanilla teacher models

# python train_teacher.py --model wrn_40_2

# python train_teacher.py --model resnet56

# python train_teacher.py --model resnet110

# CUDA_VISIBLE_DEVICES=4 python train_teacher.py --model resnet32x4 --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --resume --learning_rate 0.001 --lr_decay_epochs 30,60,100 --trial 3 --mode G --save_freq 2

# python train_teacher.py --model vgg13

# CUDA_VISIBLE_DEVICES=3 python train_ebm.py --model_s resnet26x10 --trial 6_py --capcitiy 10000 --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --g_steps 30 --save_freq 10 --epochs 500 --lr_decay_epochs_ebm 100,250 --plot_cond --learning_rate_ebm 0.0001 --batch_size 256  --energy mcmc --warmup_iters 1000 --lmda_p_x_y 1. --lmda_p_x 0 --step_size 2 --act swish --num_workers 4 --act swish --step_size 2

CUDA_VISIBLE_DEVICES=2 python train_ebm.py --model_s resnet20x10 --trial 6_py --capcitiy 10000 --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --g_steps 30 --save_freq 10 --epochs 500 --lr_decay_epochs_ebm 100,250 --plot_cond --learning_rate_ebm 0.0001 --batch_size 256  --energy mcmc --warmup_iters 1000 --lmda_p_x_y 1. --lmda_p_x 0 --lmda_e 1. --step_size 2 --act swish --num_workers 4 --act swish --step_size 2 --joint --model resnet32x4