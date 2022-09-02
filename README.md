# YOWO
You Only Watch Once: A Unified CNN Architecture for Real-Time Spatiotemporal Action Localization

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

## UCF101-24 & JHMDB21
You can download **UCF24** and **JHMDB21** from the following links:

### Google Drive
For UCF24:

Link: https://drive.google.com/file/d/1Dwh90pRi7uGkH5qLRjQIFiEmMJrAog5J/view?usp=sharing

For JHMDB21: 

Link: https://drive.google.com/file/d/15nAIGrWPD4eH3y5OTWHiUbjwsr-9VFKT/view?usp=sharing

### BaiduYunDisk
For UCF24:

Link: https://pan.baidu.com/s/11GZvbV0oAzBhNDVKXsVGKg

Password: hmu6 

For JHMDB21: 

Link: https://pan.baidu.com/s/1HSDqKFWhx_vF_9x6-Hb8jA 

Password: tcjd 

## AVA
You can use instructions from [here](https://github.com/yjh0410/AVA_Dataset) to prepare **AVA** dataset.

# Experiment
## UCF24
|    Model    |    Frame mAP    |   FPS   |    Cls Accu    |    Recall    |    Weight    |
|-------------|-----------------|---------|----------------|--------------|--------------|
|    YOWO     |      80.4       |    -    |      94.5      |      93.5    |       -      |
| YOWO (Ours) |      80.4       |    36   |      94.1      |      93.7    | [github](https://github.com/yjh0410/PyTorch_YOWO/releases/download/yowo-weight/yowo_80.4.pth)   |


## AVA v2.2
|    Model    |    Clip    |    mAP    |   FPS   |    weight    |
|-------------|------------|-----------|---------|--------------|
|    YOWO     |     16     |   17.9    |    33   |       -      |
|    YOWO     |     32     |   19.1    |         |       -      |
| YOWO (Ours) |     16     |   20.6    |    33   |  [github](https://github.com/yjh0410/PyTorch_YOWO/releases/download/yowo-weight/yowo_ava_v2.2_20.6.pth)  |
