export NCCL_P2P_DISABLE=1



port_nr=1234

batch_size=64 # 8x8 x4gpu
accumulation_steps=1
ngpu=4
cuda_devices=0,1,2,3
n_frames=16
lr_=1e-05
n_clip=1
n_crop=1
val_batch_size=8
init_temperature=0.02 # INIT_TEMPERATURE
min_temperature=0.001



result_folder=to_specify # path of the saved model is $result_folder/$model_name
epoch_nr=9

for fuse_weight_for_origin in 0.2
do
  model_name=ckpt_epoch_${epoch_nr}.pth

  sub_result_folder=fuse_$fuse_weight_for_origin
  CUDA_VISIBLE_DEVICES=$cuda_devices python -m torch.distributed.launch --master_port=$port_nr --nproc_per_node=$ngpu main.py -cfg configs/zero_shot/eval/16_32_MiniSSv2_test.yaml --output $result_folder/$sub_result_folder/eval_minissv2 --only_test --resume $result_folder/$model_name --opts TEST.NUM_CLIP $n_clip TEST.NUM_CROP $n_crop DATA.NUM_FRAMES $n_frames VAL_BATCH_SIZE $val_batch_size MODEL.FUSE_WEIGHT_FOR_ORIGIN $fuse_weight_for_origin


  CUDA_VISIBLE_DEVICES=$cuda_devices python -m torch.distributed.launch --master_port=$port_nr --nproc_per_node=$ngpu main.py -cfg configs/zero_shot/eval/ucf/16_32_vifi_clip_zs_ucf101_split1.yaml --output $result_folder/$sub_result_folder/eval_ucf_split1 --only_test --resume $result_folder/$model_name --opts TEST.NUM_CLIP $n_clip TEST.NUM_CROP $n_crop DATA.NUM_FRAMES $n_frames VAL_BATCH_SIZE $val_batch_size MODEL.FUSE_WEIGHT_FOR_ORIGIN $fuse_weight_for_origin
  CUDA_VISIBLE_DEVICES=$cuda_devices python -m torch.distributed.launch --master_port=$port_nr --nproc_per_node=$ngpu main.py -cfg configs/zero_shot/eval/ucf/16_32_vifi_clip_zs_ucf101_split2.yaml --output $result_folder/$sub_result_folder/eval_ucf_split2 --only_test --resume $result_folder/$model_name --opts TEST.NUM_CLIP $n_clip TEST.NUM_CROP $n_crop DATA.NUM_FRAMES $n_frames VAL_BATCH_SIZE $val_batch_size MODEL.FUSE_WEIGHT_FOR_ORIGIN $fuse_weight_for_origin
  CUDA_VISIBLE_DEVICES=$cuda_devices python -m torch.distributed.launch --master_port=$port_nr --nproc_per_node=$ngpu main.py -cfg configs/zero_shot/eval/ucf/16_32_vifi_clip_zs_ucf101_split3.yaml --output $result_folder/$sub_result_folder/eval_ucf_split3 --only_test --resume $result_folder/$model_name --opts TEST.NUM_CLIP $n_clip TEST.NUM_CROP $n_crop DATA.NUM_FRAMES $n_frames VAL_BATCH_SIZE $val_batch_size MODEL.FUSE_WEIGHT_FOR_ORIGIN $fuse_weight_for_origin

  CUDA_VISIBLE_DEVICES=$cuda_devices python -m torch.distributed.launch --master_port=$port_nr --nproc_per_node=$ngpu main.py -cfg configs/zero_shot/eval/hmdb/16_32_vifi_clip_zs_hmdb51_split1.yaml --output $result_folder/$sub_result_folder/eval_hmdb_split1 --only_test --resume $result_folder/$model_name --opts TEST.NUM_CLIP $n_clip TEST.NUM_CROP $n_crop DATA.NUM_FRAMES $n_frames VAL_BATCH_SIZE $val_batch_size MODEL.FUSE_WEIGHT_FOR_ORIGIN $fuse_weight_for_origin
  CUDA_VISIBLE_DEVICES=$cuda_devices python -m torch.distributed.launch --master_port=$port_nr --nproc_per_node=$ngpu main.py -cfg configs/zero_shot/eval/hmdb/16_32_vifi_clip_zs_hmdb51_split2.yaml --output $result_folder/$sub_result_folder/eval_hmdb_split2 --only_test --resume $result_folder/$model_name --opts TEST.NUM_CLIP $n_clip TEST.NUM_CROP $n_crop DATA.NUM_FRAMES $n_frames VAL_BATCH_SIZE $val_batch_size MODEL.FUSE_WEIGHT_FOR_ORIGIN $fuse_weight_for_origin
  CUDA_VISIBLE_DEVICES=$cuda_devices python -m torch.distributed.launch --master_port=$port_nr --nproc_per_node=$ngpu main.py -cfg configs/zero_shot/eval/hmdb/16_32_vifi_clip_zs_hmdb51_split3.yaml --output $result_folder/$sub_result_folder/eval_hmdb_split3 --only_test --resume $result_folder/$model_name --opts TEST.NUM_CLIP $n_clip TEST.NUM_CROP $n_crop DATA.NUM_FRAMES $n_frames VAL_BATCH_SIZE $val_batch_size MODEL.FUSE_WEIGHT_FOR_ORIGIN $fuse_weight_for_origin

  CUDA_VISIBLE_DEVICES=$cuda_devices python -m torch.distributed.launch --master_port=$port_nr --nproc_per_node=$ngpu main.py -cfg configs/zero_shot/eval/k600/16_32_K600_ZS_split1.yaml --output $result_folder/$sub_result_folder/eval_k600_split1 --only_test --resume $result_folder/$model_name --opts TEST.NUM_CLIP $n_clip TEST.NUM_CROP $n_crop DATA.NUM_FRAMES $n_frames VAL_BATCH_SIZE $val_batch_size MODEL.FUSE_WEIGHT_FOR_ORIGIN $fuse_weight_for_origin
  CUDA_VISIBLE_DEVICES=$cuda_devices python -m torch.distributed.launch --master_port=$port_nr --nproc_per_node=$ngpu main.py -cfg configs/zero_shot/eval/k600/16_32_K600_ZS_split2.yaml --output $result_folder/$sub_result_folder/eval_k600_split2 --only_test --resume $result_folder/$model_name --opts TEST.NUM_CLIP $n_clip TEST.NUM_CROP $n_crop DATA.NUM_FRAMES $n_frames VAL_BATCH_SIZE $val_batch_size MODEL.FUSE_WEIGHT_FOR_ORIGIN $fuse_weight_for_origin
  CUDA_VISIBLE_DEVICES=$cuda_devices python -m torch.distributed.launch --master_port=$port_nr --nproc_per_node=$ngpu main.py -cfg configs/zero_shot/eval/k600/16_32_K600_ZS_split3.yaml --output $result_folder/$sub_result_folder/eval_k600_split3 --only_test --resume $result_folder/$model_name --opts TEST.NUM_CLIP $n_clip TEST.NUM_CROP $n_crop DATA.NUM_FRAMES $n_frames VAL_BATCH_SIZE $val_batch_size MODEL.FUSE_WEIGHT_FOR_ORIGIN $fuse_weight_for_origin


  CUDA_VISIBLE_DEVICES=$cuda_devices python -m torch.distributed.launch --master_port=$port_nr --nproc_per_node=$ngpu main.py -cfg configs/zero_shot/eval/16_32_Charades_test.yaml --output $result_folder/$sub_result_folder/eval_charades --only_test --resume $result_folder/$model_name --opts TEST.NUM_CLIP $n_clip TEST.NUM_CROP $n_crop DATA.NUM_FRAMES $n_frames VAL_BATCH_SIZE $val_batch_size MODEL.FUSE_WEIGHT_FOR_ORIGIN $fuse_weight_for_origin

  CUDA_VISIBLE_DEVICES=$cuda_devices python -m torch.distributed.launch --master_port=$port_nr --nproc_per_node=$ngpu main.py -cfg configs/zero_shot/eval/UAV_human/16_32_vifi_clip_zs_uavhuman_split1.yaml --output $result_folder/$sub_result_folder/eval_uavhuman_split1 --only_test --resume $result_folder/$model_name --opts TEST.NUM_CLIP $n_clip TEST.NUM_CROP $n_crop DATA.NUM_FRAMES $n_frames VAL_BATCH_SIZE $val_batch_size MODEL.FUSE_WEIGHT_FOR_ORIGIN $fuse_weight_for_origin
  CUDA_VISIBLE_DEVICES=$cuda_devices python -m torch.distributed.launch --master_port=$port_nr --nproc_per_node=$ngpu main.py -cfg configs/zero_shot/eval/UAV_human/16_32_vifi_clip_zs_uavhuman_split2.yaml --output $result_folder/$sub_result_folder/eval_uavhuman_split2 --only_test --resume data/$result_folder/$model_name --opts TEST.NUM_CLIP $n_clip TEST.NUM_CROP $n_crop DATA.NUM_FRAMES $n_frames VAL_BATCH_SIZE $val_batch_size MODEL.FUSE_WEIGHT_FOR_ORIGIN $fuse_weight_for_origin

  CUDA_VISIBLE_DEVICES=$cuda_devices python -m torch.distributed.launch --master_port=$port_nr --nproc_per_node=$ngpu main.py -cfg configs/zero_shot/eval/16_32_vifi_clip_zs_Momentsintime_test.yaml --output $result_folder/$sub_result_folder/eval_momentsintime --only_test --resume $result_folder/$model_name --opts TEST.NUM_CLIP $n_clip TEST.NUM_CROP $n_crop DATA.NUM_FRAMES $n_frames VAL_BATCH_SIZE $val_batch_size MODEL.FUSE_WEIGHT_FOR_ORIGIN $fuse_weight_for_origin

  python utils/eval_summary.py -dir=$result_folder/$sub_result_folder
done

