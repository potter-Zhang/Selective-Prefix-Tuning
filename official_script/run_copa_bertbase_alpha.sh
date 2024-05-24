export TASK_NAME=superglue
export DATASET_NAME=copa
export CUDA_VISIBLE_DEVICES=0

bs=16
lr=1e-2
dropout=0.1
psl=20
epoch=50
alpha=1e-4
tau=8

python3 run.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-bert-base/ultimate \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --selective_prefix \
  --tau $tau \
  --alpha $alpha \
  --scores_dropout_prob 0.3 \
  --overwrite_cache 