# Train YOWOF-R18
python train.py \
        --cuda \
        -d jhmdb21 \
        -v yowo \
        --num_workers 4 \
        --eval_epoch 1 \
        --eval \
