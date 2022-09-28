# YOWO
Big thanks to [YOWO](https://github.com/wei-tim/YOWO) for their open source. I reimplemented ```YOWO``` and reproduced the performance. On the ```AVA``` dataset, my reproduced YOWO is better than the official YOWO. I hope that such a real-time action detector with simple structure and superior performance can attract your interest in the task of spatio-temporal action detection.

# Requirements
- We recommend you to use Anaconda to create a conda environment:
```Shell
conda create -n yowo python=3.6
```

- Then, activate the environment:
```Shell
conda activate yowo
```

- Requirements:
```Shell
pip install -r requirements.txt 
```

# Dataset
You can download **UCF24** and **JHMDB21** from the following links:

## UCF101-24:
* Google drive

Link: https://drive.google.com/file/d/1Dwh90pRi7uGkH5qLRjQIFiEmMJrAog5J/view?usp=sharing

* BaiduYun Disk

Link: https://pan.baidu.com/s/11GZvbV0oAzBhNDVKXsVGKg

Password: hmu6 

## JHMDB21: 
* Google drive

Link: https://drive.google.com/file/d/15nAIGrWPD4eH3y5OTWHiUbjwsr-9VFKT/view?usp=sharing

* BaiduYun Disk

Link: https://pan.baidu.com/s/1HSDqKFWhx_vF_9x6-Hb8jA 

Password: tcjd 

## AVA
You can use instructions from [here](https://github.com/yjh0410/AVA_Dataset) to prepare **AVA** dataset.

# Experiment
## UCF24
|    Model    |  Clip  |Frame mAP | Video mAP |   FPS   |    Weight    |
|-------------|--------|----------|-----------|---------|--------------|
|    YOWO     |   16   |   80.4   |   48.8    |    -    |       -      |
| YOWO (Ours) |   16   |   84.9   |       |    36   | [github](https://github.com/yjh0410/PyTorch_YOWO/releases/download/yowo-weight/yowo_80.4.pth)   |

Our SOTA results on UCF24:

```Shell
AP: 85.25% (1)
AP: 96.94% (10)
AP: 78.58% (11)
AP: 68.61% (12)
AP: 78.98% (13)
AP: 94.92% (14)
AP: 90.00% (15)
AP: 77.44% (16)
AP: 75.82% (17)
AP: 91.07% (18)
AP: 97.16% (19)
AP: 62.71% (2)
AP: 93.22% (20)
AP: 79.16% (21)
AP: 80.07% (22)
AP: 76.10% (23)
AP: 92.49% (24)
AP: 86.29% (3)
AP: 76.99% (4)
AP: 74.89% (5)
AP: 95.74% (6)
AP: 93.68% (7)
AP: 93.71% (8)
AP: 97.13% (9)
mAP: 84.87%
```

## AVA v2.2
|    Model    |    Clip    |    mAP    |   FPS   |    weight    |
|-------------|------------|-----------|---------|--------------|
|    YOWO     |     16     |   17.9    |    33   |       -      |
|    YOWO     |     32     |   19.1    |         |       -      |
| YOWO (Ours) |     16     |   20.6    |    33   |  [github](https://github.com/yjh0410/PyTorch_YOWO/releases/download/yowo-weight/yowo_ava_v2.2_20.6.pth)  |

## Train YOWO
* UCF101-24

```Shell
python train.py --cuda -d ucf24 -v yowo --num_workers 4 --eval_epoch 1 --eval
```

or you can just run the script:

```Shell
sh train_ucf.sh
```

* AVA
```Shell
python train.py --cuda -d ava_v2.2 -v yowo --num_workers 4 --eval_epoch 1 --eval
```

or you can just run the script:

```Shell
sh train_ava.sh
```

##  Test YOWO
* UCF101-24

```Shell
python test.py --cuda -d ucf24 -v yowo --weight path/to/weight --show
```

* AVA

```Shell
python test.py --cuda -d ava_v2.2 -v yowo --weight path/to/weight --show
```

## Evaluate YOWO
* UCF101-24

The detection results have been saved after each epoch of training phase, so you can 
run the following command to calculate mAP. For example, if you want calculate the mAP
of YOWO trained for 2 epochs:

```Shell
python eval.py \
        --cuda \
        -d ucf24 \
        -v yowo \
        -bs 8 \
        -size 224 \
        --gt_folder ./evaluator/groundtruths_ucf_jhmdb/groundtruths_ucf/ \
        --dt_folder ./results/ucf_detections/yowo/detections_2/ \
        --save_path ./evaluator/eval_results/ \
        --weight path/to/weight \
        --cal_mAP \
```

If you want to evaluate from scratch, you can run the following command to recalucate
`classification accuracy`, `recall` and `mAP`:

```Shell
python eval.py \
        --cuda \
        -d ucf24 \
        -v yowo \
        --gt_folder ./evaluator/groundtruths_ucf_jhmdb/groundtruths_ucf/ \
        --weight path/to/weight \
        --cal_mAP \
        --redo
```

* AVA

Run the following command to calculate ```mAP```:

```Shell
python eval.py \
        --cuda \
        -d ava_v2.2 \
        -v yowo \
        --weight path/to/weight
```
