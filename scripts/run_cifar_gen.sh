# sample scripts for training vanilla teacher models

# python train_teacher.py --model wrn_40_2

# python train_teacher.py --model resnet56

# python train_teacher.py --model resnet110

# CUDA_VISIBLE_DEVICES=4 python train_teacher.py --model resnet32x4 --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --resume --learning_rate 0.001 --lr_decay_epochs 30,60,100 --trial 3 --mode G --save_freq 2

# python train_teacher.py --model vgg13

CUDA_VISIBLE_DEVICES=5 python train_ebm.py --model_s resnet32x10 --trial 3 --capcitiy 10000 --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --g_steps 20 --save_freq 10 --epochs 500 --lr_decay_epochs 100,250 --plot_cond --learning_rate 0.00001 --batch_size 128  --energy mcmc --warmup_iters 1000 --lmda_p_x_y 1. --lmda_p_x 0 --step_size 1 --act swish --num_workers 4 --data_noise 0.03 --load_buffer_path /data/lijingru/EBKD/save/student_model/resnet32x10_cifar100_lr_0.0001_decay_0.0_buffer_size_10000_lpx_0.0_lpxy_1.0_energy_mode_mcmc_step_size_1.0_trial_3_cls_mode_cls/res_epoch_180.pts --init_epochs 180