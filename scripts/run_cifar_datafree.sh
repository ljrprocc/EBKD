# CUDA_VISIBLE_DEVICES=0 python train_student.py --batch_size 128 --model_s resnet8x4  --distill energy -r 0.1 -a 0.9 -b 0.1 --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth  --trial 3_ssm --num_workers 4
# CUDA_VISIBLE_DEVICES=2 python train_student.py --batch_size 128 --model_s resnet8x4  --distill ebkd -r 0 -a 0 -b 1 --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth  --trial 1 --num_workers 4 --learning_rate 0.05 --epochs 300
for (( i=0; i<1; i=i+1 )); do
echo "**************** $i th training**************"
CUDA_VISIBLE_DEVICES=5 python train_student.py --batch_size 128 --model_s resnet8x4  --distill kd -r 0.3 -a 0.7 -b 0 --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth  --trial ${i}_eval_40000_lc_0_2 --num_workers 4 --learning_rate_ebm 0.01 --epochs 300 --datafree --norm batch --df_folder /data/lijingru/EBKD/save/student_model/resnet28x10_cifar100_lr_0.0001_decay_0.0_buffer_size_40000_trial_fresh_deepinv_5_noenergy_epoch_3001_gsteps_40_step_size/img_sample_eval/
done
# /data/lijingru/EBKD/save/student_model/resnet28x10_T:resnet32x4_S:resnet8x4_cifar100_lr_0.0001_decay_0.0_buffer_size_10000_lpx_1.0_lpxy_0.0_energy_mode_mcmc_step_size_1.0_trial_2_st_10_cls_mode_cls/
