# sample scripts for training vanilla teacher models

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="0,4,5,6" python train_ebm.py --model_s ResNet50cifar100 --trial 1 --capcitiy 10000 --g_steps 20 --save_freq 20 --epochs 200 --lr_decay_epochs 70,150 --plot_cond --learning_rate 0.0001 --batch_size 128  --energy mcmc --warmup_iters 1000 --lmda_p_x_y 0. --lmda_p_x 1. --step_size 5. --act swish --num_workers 8 --data_noise 0.03 --print_freq 100 --use_py --reinit_freq 0.05 --print_freq 100 --multiscale --dataset imagenet --world_size 4
