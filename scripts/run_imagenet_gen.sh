# sample scripts for training vanilla teacher models

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="1,2,3,6" python train_ebm.py --model_s ResNet50 --trial 1 --capcitiy 10000 --g_steps 20 --save_freq 20 --epochs 200 --lr_decay_epochs 70,150 --plot_uncond --learning_rate 0.0001 --batch_size 64  --energy mcmc --warmup_iters 1500 --lmda_p_x_y 0. --lmda_p_x 1. --step_size 1. --act swish --num_workers 8 --data_noise 0.03 --print_freq 100 --reinit_freq 0.02 --print_freq 100 --dataset imagenet --world_size 4 --multiscale
