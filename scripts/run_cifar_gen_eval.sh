CUDA_VISIBLE_DEVICES=0 python eval_ebm.py --model_s resnet28x10 --trial fresh_3 --buffer_size 40000 --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --save_freq 5 --epochs 150 --plot_cond --learning_rate 0.0001 --batch_size 100  --energy mcmc --warmup_iters 1000 --lmda_p_x_y 1. --lmda_p_x 0. --load_buffer_path /data/lijingru/EBKD/save/student_model/res_epoch_84.pts --n_sample_steps 5000 --g_steps 100 --open_debug --print_every 200 --dataset cifar100 --fresh --step_size 1. --resume /data/lijingru/EBKD/save/student_model/resnet28x10_cifar100_lr_0.0001_decay_0.0_buffer_size40000_lpx_0.0_lpxy_1.0_energy_mode_mcmc_trial_fresh_5000_g_100/img_ckpts/res_buffer_1600.pts --init_epoch 1600