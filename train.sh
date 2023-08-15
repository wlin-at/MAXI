export NCCL_P2P_DISABLE=1


output_dir=to_specify

port_nr=1234
cuda_devices=0,1,2,3
n_frames=16
batch_size=16 # 8x8 x4gpu
accumulation_steps=4
ngpu=4
lr_=5e-06
n_clip=4
n_crop=3
val_batch_size=8
init_temperature=0.02 # INIT_TEMPERATURE
min_temperature=0.001
save_freq=1

use_description_type='train cls+cap mil nce'
bag_type='gpt3verb+blipwords'
n_samples_in_bag=16
caption_bag_dir=data/caption_verbs_bag
gpt3_bag_dir=data/gpt3_verbs

train_file=datasets_splits/k400_splits/clip_match_result_thresh0.9.txt

CUDA_VISIBLE_DEVICES=$cuda_devices python -m torch.distributed.launch --master_port=$port_nr --nproc_per_node=$ngpu main.py -cfg configs/zero_shot/train/k400/maxi.yaml --output $output_dir --opts DATA.NUM_FRAMES $n_frames TRAIN.LR $lr_ TRAIN.BATCH_SIZE $batch_size TRAIN.ACCUMULATION_STEPS $accumulation_steps MODEL.INIT_TEMPERATURE $init_temperature MODEL.MIN_TEMPERATURE $min_temperature SAVE_FREQ $save_freq DATA.USE_DESCRIPTION_TYPE "$use_description_type" DATA.CAPTION_BAG_DIR $caption_bag_dir DATA.GPT3_BAG_DIR $gpt3_bag_dir DATA.N_SAMPLES_IN_BAG $n_samples_in_bag DATA.BAG_TYPE $bag_type DATA.TRAIN_FILE $train_file




