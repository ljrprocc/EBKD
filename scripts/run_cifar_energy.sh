# CUDA_VISIBLE_DEVICES=0 python train_student.py --batch_size 128 --model_s resnet8x4  --distill energy -r 0.1 -a 0.9 -b 0.1 --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth  --trial 3_ssm --num_workers 4
# CUDA_VISIBLE_DEVICES2 python train_student.py --batch_size 128 --model_s resnet8x4  --distill ebkd -r 0 -a 0 -b 1 --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth  --trial 1 --num_workers 4 --learning_rate 0.05 --epochs 300
for (( i=0; i<10; i=i+1 )); do
echo "**************** $i th training**************"
CUDA_VISIBLE_DEVICES=4 python train_student.py --batch_size 128 --model_s wrn_16_1  --distill hint -a 0 -b 100 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth  --trial ${i}_ebm --num_workers 4 --learning_rate 0.01 --epochs 300 --G_step 1 --mode energy
done
