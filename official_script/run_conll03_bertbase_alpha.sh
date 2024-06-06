export TASK_NAME=ner
export DATASET_NAME=conll2003
export CUDA_VISIBLE_DEVICES=1

bs=16
epoch=60
psl=11
lr=3e-2
dropout=0.1
tau=8
alpha=1e-4

python3 run.py \
  --model_name_or_path bert-base-uncased \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 152 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-bert-large/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --selective_prefix \
  --tau $tau \
  --alpha $alpha \
  --overwrite_cache