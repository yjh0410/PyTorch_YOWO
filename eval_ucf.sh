python eval.py \
        --cuda \
        -d ucf24 \
        -v yowo-d19 \
        -bs 8 \
        -size 224 \
        --gt_folder ./evaluator/groundtruths_ucf_jhmdb/groundtruths_ucf/ \
        --dt_folder ./results/ucf_detections/detections_1/ \
        --save_path ./evaluator/eval_results/ \
        --weight ./weights/ucf24/yowo-d19/yowof-d19_80.4.pth \
        --cal_mAP \
        # --redo \
