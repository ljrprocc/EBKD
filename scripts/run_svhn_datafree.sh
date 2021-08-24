for (( i=0; i<10; i=i+1 )); do
echo "**************** $i th training**************"
CUDA_VISIBLE_DEVICES=1 python train_student.py --batch_size 128 --model_s resnet8x4  --distill kd -r 0.4 -a 0.6 -b 0 --path_t /data/lijingru/EBKD/save/models/resnet32x4_svhn_lr_0.01_decay_0.0005_trial_0/resnet32x4_best.pth  --trial ${i}_k_10 --num_workers 4 --learning_rate_ebm 0.01 --epochs 300 --datafree --norm batch --df_folder /data/lijingru/EBKD/save/student_model/resnet28x10_T:resnet32x4_S:resnet8x4_svhn_lr_0.0001_decay_0.0_buffer_size_10000_lpx_1.0_lpxy_0.0_energy_mode_mcmc_step_size_1.0_trial_1_st_random_cls_mode_cls_k_10/img_sample_eval/ --dataset svhn
done
