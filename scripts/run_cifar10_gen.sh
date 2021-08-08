# CUDA_VISIBLE_DEVICES=3 python train_ebm.py --model_s resnet26x10 --trial 6_py --capcitiy 10000 --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --g_steps 30 --save_freq 10 --epochs 500 --lr_decay_epochs_ebm 100,250 --plot_cond --learning_rate_ebm 0.0001 --atch_size 256  --energy mcmc --warmup_iters 1000 --lmda_p_x_y 1. --lmda_p_x 0 --step_size 2 --act swish --num_workers 4 --act swish --step_size 2

# CUDA_VISIBLE_DEVICES=0 python train_ebm.py --model_s resnet20x10 --trial 8_lc --capcitiy 10000 --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --g_steps 20 --save_freq 10 --epochs 500 --lr_decay_epochs_ebm 100,250 --plot_cond --learning_rate_ebm 0.0001 --batch_size 128  --energy mcmc --warmup_iters 1000 --lmda_p_x_y 1. --lmda_p_x 0 --lmda_e 1. --step_size 1 --act swish --num_workers 4 --act swish --model resnet32x4 --joint --model_stu resnet8x4

CUDA_VISIBLE_DEVICES=2 python train_ebm.py --model_s resnet8x4 --trial 1 --capcitiy 10000 --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --g_steps 20 --save_freq 10 --epochs 150 --lr_decay_epochs 50,100 --plot_uncond --learning_rate 0.0001 --batch_size 128  --energy mcmc --warmup_iters 1000 --lmda_p_x_y 0. --lmda_p_x 1. --step_size 1. --act swish --num_workers 4 --data_noise 0.03 --print_freq 100 --dataset cifar10