# Train YOWO
python train.py \
        --cuda \
        -d ucf24 \
        -v yowo_nano \
        --num_workers 4 \
        --eval_epoch 1 \
        # --eval \
        # --fp16 \
