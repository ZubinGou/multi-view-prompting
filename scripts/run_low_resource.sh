set -ex

export CUDA_VISIBLE_DEVICES=0

cd src

TASK=aste
for DATA in laptop14
do
for DATA_RATIO in 0.01 0.02 0.05 0.1 0.2 # few-shot
# for DATA_RATIO in 0.0 # zero-shot
do
for SEED in 5 10 15 20 25
do
for K in 5
# for K in 1 3 7 15
do
INFER_PATH=$K
CTRL_TOKEN=post
OUT_DIR="../outputs/$TASK/${DATA}/top${K}_${CTRL_TOKEN}_data${DATA_RATIO}_acos5_seed${SEED}"

mkdir -p $OUT_DIR

python main.py \
    --data_path "../data/" \
    --dataset $DATA \
    --model_name_or_path outputs/acos/rest16/top5_post_data1.0_seed${SEED}/final \
    --output_dir $OUT_DIR \
    --num_train_epochs 20 \
    --save_top_k 0 \
    --task $TASK \
    --top_k $K \
    --ctrl_token $CTRL_TOKEN \
    --multi_path \
    --num_path $INFER_PATH \
    --seed $SEED \
    --train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lowercase \
    --sort_label \
    --data_ratio $DATA_RATIO \
    --check_val_every_n_epoch 10  \
    --agg_strategy vote \
    --eval_batch_size 64 \
    --constrained_decode \
    --do_train \
    # --load_path_cache \
    # > $OUT_DIR/train.log 2>&1&
done
done
done
done
