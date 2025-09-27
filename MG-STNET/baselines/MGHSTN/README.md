# MGHSTN

**CIKM‘24** Urban Traffic Accident Risk Prediction Revisited: Regionality, Proximity, Similarity and Sparsity

**M**ulti-**G**ranularity **H**ierarchical **S**patio-**T**emporal **N**etwork

## Environment Requirement

The code runs well under python 3.11.5. The required packages are as follows:

- pytorch==2.0.1
- torchvision==0.15.2
- numpy==1.25.2
- pickle, argparse, matplotlib, pandas

## Setup

Unzip datasets:

```
unzip chicago.zip
unzip chicago_remote_sensing_256.zip
unzip nyc.zip
unzip nyc_remote_sensing_256.zip
```

## Usage

train model on NYC:
```
python train.py --config config/nyc/NYC_Config.json --gpus 0
```


train model on Chicago:
```
python train.py --config config/chicago/Chicago_Config.json --gpus 0
```


## Citation

```
@inproceedings{Chen2024mghstn,
  author       = {Minxiao Chen and
                  Haitao Yuan and
                  Nan Jiang and
                  Zhifeng Bao and
                  Shangguang Wang},
  editor       = {Edoardo Serra and
                  Francesca Spezzano},
  title        = {Urban Traffic Accident Risk Prediction Revisited: Regionality, Proximity,
                  Similarity and Sparsity},
  booktitle    = {CIKM},
  pages        = {281--290},
  publisher    = {{ACM}},
  year         = {2024},
}
```