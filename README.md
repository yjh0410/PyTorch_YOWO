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
You can download **UCF24** and **JHMDB21** from the following link:

For UCF24:

Link: https://pan.baidu.com/s/11GZvbV0oAzBhNDVKXsVGKg

Password: hmu6 

For JHMDB21: 

Link: https://pan.baidu.com/s/1HSDqKFWhx_vF_9x6-Hb8jA 

Password: tcjd 


# Experiment
## UCF24
|    Model    |    Frame mAP    |    Cls Accu    |    Recall    |    Weight    |
|-------------|-----------------|----------------|--------------|--------------|
|    YOWO     |      80.4       |      94.5      |      93.5    |       -      |
| YOWO (Ours) |      80.4       |      94.1      |      93.7    | [github](https://github.com/yjh0410/PyTorch_YOWO/releases/download/yowo-weight/yowof-d19_80.4.pth)   |