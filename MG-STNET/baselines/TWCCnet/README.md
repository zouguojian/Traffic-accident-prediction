# TWCCnet

This repository is developed on top of the code provided by GSNet authors correspoding to AAAI 2021. GSNet: Learning Spatial-Temporal Correlations from Geographical and Semantic Aspects for Traffic Accident Risk Forecasting and therefore, we are thankful for the support.

# Commands

Unzip data.zip

## NYC dataset

```
python3 train.py --config config/nyc/TWCCNet_NYC_Config.json --gpus 0
```

## Chicago dataset

```
python3 train.py --config config/chicago/TWCCNet_Chicago_Config.json --gpus 0
```
