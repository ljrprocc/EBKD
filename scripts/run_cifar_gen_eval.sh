python eval_ebm.py --model_s resnet28x10 --trial fresh_deepinv_5_noenergy --capcitiy 40000 --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --path_s /data/lijingru/EBKD/save/models/resnet8x4_cifar100_lr_0.01_decay_0.0005_trial_0/resnet8x4_best.pth --save_freq 5 --epochs 3001 --learning_rate 0.0001 --batch_size 256  --energy mcmc --warmup_iters 1000 --gpu 3 --norm batch --step_size 0.2 --load_buffer_path /data/lijingru/EBKD/save/student_model/cifar10_lr_0.002_decay_0.0_ndf_256_trial_1_resnet28x10_cifar10_lr_0.0003_decay_0_buffer_size_40000_lpx_1.0_lpxy_0.0_energy_mode_mcmc_step_size_1_g_steps_20_trial_1/res_epoch_300.pts --init_epochs 300 --reinit_freq 0. --lmda_lc 0.2 --dataset cifar10