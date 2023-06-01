#/bin/bash
set -ex


cd src


for DATA in rest15
do

TASK=asqp
K=1
INFER_PATH=$K
CTRL_TOKEN=none
OUT_DIR="../outputs/$TASK/${DATA}/top${K}_${CTRL_TOKEN}_data${DATA_RATIO}_seed${SEED}"

mkdir -p $OUT_DIR

python -u llms/infer.py \
    --data_path "../data/" \
    --dataset $DATA \
    --model_name_or_path t5-base \
    --output_dir $OUT_DIR \
    --save_top_k 0 \
    --task $TASK \
    --top_k $K \
    --ctrl_token $CTRL_TOKEN \
    --num_path $INFER_PATH \
    --seed 0 \
    --train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lowercase \
    --sort_label \
    --check_val_every_n_epoch 10 \
    --constrained_decode \
    --single_view_type heuristic \

done