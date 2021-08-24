# sample scripts for training vanilla teacher models

# python train_teacher.py --model wrn_40_2

# python train_teacher.py --model resnet56

# python train_teacher.py --model resnet110

# CUDA_VISIBLE_DEVICES=4 python train_teacher.py --model resnet32x4 --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --resume --learning_rate 0.001 --lr_decay_epochs 30,60,100 --trial 3 --mode G --save_freq 2

# python train_teacher.py --model vgg13

# CUDA_VISIBLE_DEVICES=3 python train_ebm.py --model_s resnet26x10 --trial 6_py --capcitiy 10000 --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --g_steps 30 --save_freq 10 --epochs 500 --lr_decay_epochs_ebm 100,250 --plot_cond --learning_rate_ebm 0.0001 --atch_size 256  --energy mcmc --warmup_iters 1000 --lmda_p_x_y 1. --lmda_p_x 0 --step_size 2 --act swish --num_workers 4 --act swish --step_size 2

# CUDA_VISIBLE_DEVICES=0 python train_ebm.py --model_s resnet20x10 --trial 8_lc --capcitiy 10000 --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --g_steps 20 --save_freq 10 --epochs 500 --lr_decay_epochs_ebm 100,250 --plot_cond --learning_rate_ebm 0.0001 --batch_size 128  --energy mcmc --warmup_iters 1000 --lmda_p_x_y 1. --lmda_p_x 0 --lmda_e 1. --step_size 1 --act swish --num_workers 4 --act swish --model resnet32x4 --joint --model_stu resnet8x4

CUDA_VISIBLE_DEVICES=1 python train_ebm.py --model_s resnet28x10 --trial new_random --capcitiy 10000 --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --g_steps 20 --save_freq 2 --epochs 120 --lr_decay_epochs 50,84 --plot_cond --learning_rate 0.00002 --batch_size 64  --energy mcmc --warmup_iters 1000 --lmda_p_x_y 0. --lmda_p_x 1. --step_size 1. --act leaky --num_workers 4 --data_noise 0.03 --print_freq 100 --use_py --joint --model resnet32x4 --model_stu resnet8x4 --lc_K 10 --st -1 --load_buffer_path /data/lijingru/EBKD/save/student_model/resnet28x10_T:resnet32x4_S:resnet8x4_cifar100_lr_0.0001_decay_0.0_buffer_size_10000_lpx_1.0_lpxy_0.0_energy_mode_mcmc_step_size_1.0_trial_new_random_cls_mode_cls_k_10/res_epoch_84.pts --init_epochs 84
