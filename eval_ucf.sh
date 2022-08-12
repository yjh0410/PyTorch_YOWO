python eval.py \
        --cuda \
        -d ucf24 \
        -v yowo \
        -bs 8 \
        -size 224 \
        --gt_folder ./evaluator/groundtruths_ucf_jhmdb/groundtruths_ucf/ \
        --dt_folder ./results/ucf_detections/detections_5/ \
        --save_path ./evaluator/eval_results/ \
        --weight ./weights/ucf24/yowo/yowo_epoch_5_92.8_95.0.pth \
        --cal_mAP \
        # --redo \
