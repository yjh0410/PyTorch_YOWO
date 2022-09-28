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
| YOWO (Ours) |   16   |   84.9   |   50.5    |    36   | [github](https://github.com/yjh0410/PyTorch_YOWO/releases/download/yowo-weight/yowo_84.9.pth)   |

Our SOTA results on UCF24:


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

```Shell
# Frame mAP
python eval.py \
        --cuda \
        -d ucf24 \
        -v yowo \
        -bs 8 \
        -size 224 \
        --weight path/to/weight \
        --save_path ./evaluator/eval_results/ \
        --cal_frame_mAP \
```

Our sota result of frame mAP@0.5 IoU on UCF101-24:
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

```Shell
# Video mAP
python eval.py \
        --cuda \
        -d ucf24 \
        -v yowo \
        -bs 8 \
        -size 224 \
        --weight path/to/weight \
        --save_path ./evaluator/eval_results/ \
        --cal_video_mAP \
```

Our sota result of video mAP@0.5 IoU on UCF101-24:
```Shell
-------------------------------
V-mAP @ 0.05 IoU:
--Per AP:  [94.1, 99.64, 68.62, 97.44, 87.21, 100.0, 82.72, 100.0, 99.87, 96.08, 44.8, 92.43, 91.76, 100.0, 24.29, 92.53, 90.23, 96.55, 94.24, 63.46, 73.44, 51.48, 82.85, 88.67]
--mAP:  83.85
-------------------------------
V-mAP @ 0.1 IoU:
--Per AP:  [94.1, 97.37, 67.16, 97.44, 85.2, 100.0, 82.72, 100.0, 99.87, 96.08, 44.8, 92.43, 91.76, 100.0, 24.29, 92.53, 90.23, 96.55, 94.24, 63.46, 70.75, 51.48, 79.44, 88.67]
--mAP:  83.36
-------------------------------
V-mAP @ 0.2 IoU:
--Per AP:  [70.0, 97.37, 62.86, 89.47, 59.5, 100.0, 78.04, 100.0, 90.74, 96.08, 44.8, 92.43, 91.76, 100.0, 22.29, 92.53, 90.23, 96.55, 94.24, 58.8, 42.35, 48.03, 53.41, 88.67]
--mAP:  77.51
-------------------------------
V-mAP @ 0.3 IoU:
--Per AP:  [14.33, 48.86, 61.27, 76.36, 12.58, 87.34, 78.04, 100.0, 90.74, 93.28, 44.8, 89.89, 91.76, 100.0, 15.41, 92.53, 88.99, 96.55, 94.24, 51.4, 24.52, 42.89, 5.63, 78.64]
--mAP:  65.84
-------------------------------
V-mAP @ 0.5 IoU:
--Per AP:  [0.18, 1.9, 58.16, 33.87, 1.31, 44.26, 49.09, 100.0, 61.3, 91.23, 44.8, 70.06, 59.22, 100.0, 3.73, 92.53, 87.71, 89.53, 91.29, 45.06, 0.97, 20.94, 0.0, 65.41]
--mAP:  50.52
-------------------------------
V-mAP @ 0.75 IoU:
--Per AP:  [0.0, 0.0, 27.05, 0.0, 0.0, 0.56, 9.81, 69.56, 14.42, 31.74, 3.43, 29.46, 0.93, 48.21, 0.71, 61.32, 45.81, 16.04, 84.41, 14.2, 0.06, 0.96, 0.0, 35.95]
--mAP:  20.61
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

Our sota result of frame mAP@0.5 IoU on AVA-v2.2:
```Shell


```
