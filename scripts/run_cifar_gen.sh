# sample scripts for training vanilla teacher models

# python train_teacher.py --model wrn_40_2

# python train_teacher.py --model resnet56

# python train_teacher.py --model resnet110

# CUDA_VISIBLE_DEVICES=4 python train_teacher.py --model resnet32x4 --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --resume --learning_rate 0.001 --lr_decay_epochs 30,60,100 --trial 3 --mode G --save_freq 2

# CUDA_VISIBLE_DEVICES=0 python train_ebm.py --model_s resnet20x10 --trial 8_lc --capcitiy 10000 --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --g_steps 20 --save_freq 10 --epochs 500 --lr_decay_epochs_ebm 100,250 --plot_cond --learning_rate_ebm 0.0001 --batch_size 128  --energy mcmc --warmup_iters 1000 --lmda_p_x_y 1. --lmda_p_x 0 --lmda_e 1. --step_size 1 --act swish --num_workers 4 --act swish --model resnet32x4 --joint --model_stu resnet8x4

# CUDA_VISIBLE_DEVICES=2 python train_ebm.py --model_s wrn_22_10 --trial cor_multi_5 --capcitiy 10000 --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --g_steps 20 --save_freq 5 --epochs 150 --lr_decay_epochs 50,100 --plot_cond --learning_rate 0.0001 --batch_size 128  --energy mcmc --warmup_iters 1000 --lmda_p_x_y 1. --lmda_p_x 0. --step_size 2.5 --act swish --num_workers 4 --data_noise 0.03 --print_freq 100 --use_py --reinit_freq 0.05 --print_freq 100 --multiscale

# CUDA_VISIBLE_DEVICES=2 python train_ebm.py --model_s wrn_22_10 --trial cor_multi_5 --save_freq 5 --epochs 150 --lr_decay_epochs 50,100 --plot_cond --learning_rate 0.0001 --batch_size 128  --energy mcmc --warmup_iters 1000 --lmda_p_x_y 1. --lmda_p_x 0 --num_workers 4 --data_noise 0.03 --print_freq 100 --use_py --reinit_freq 0.05 --multiscale

# python train_ebm.py --config ./configs/gz.yaml --trial short_run_4 --save_freq 20 --epochs 100 --batch_size 128  --warmup_iters 1000 --num_workers 4 --print_freq 20 --gpu 1

python train_ebm.py --config ./configs/vae.yaml --trial pure_vae_2 --save_freq 20 --epochs 200 --warmup_iters 1000 --num_workers 4 --print_freq 50 --gpu 0
