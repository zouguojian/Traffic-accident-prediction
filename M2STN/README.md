# Traffic- accident-risk-estimation

# M2STN
IEEE T-ITS 2023. M2STN: A Multi-Modal Spatio-Temporal Network for Traffic Accident Risk Forecasting  

# Usage

train model on NYC:
```
python train.py --config config/nyc/GSNet_NYC_Config.json --gpus 0
```
test model on NYC
```
 python train.py --config config/nyc/GSNet_NYC_Config.json --gpus 0 --test
```

train model on Chicago:
```
python train.py --config config/chicago/GSNet_Chicago_Config.json --gpus 0
```

test model on Chicago:
```
python train.py --config config/chicago/GSNet_Chicago_Config.json --gpus 0 --test
```

# Configuration

The configuration file config.json contains three parts: Data, Training and Predict:


# About

If you find this repository useful in your research, please cite the following paper:
```
@article{zou2023will,
  title={When Will We Arrive? A Novel Multi-Task Spatio-Temporal Attention Network Based on Individual Preference for Estimating Travel Time},
  author={Zou, Guojian and Lai, Ziliang and Ma, Changxi and Tu, Meiting and Fan, Jing and Li, Ye},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2023},
  publisher={IEEE}
}

@article{zou2023novel,
  title={A novel spatio-temporal generative inference network for predicting the long-term highway traffic speed},
  author={Zou, Guojian and Lai, Ziliang and Ma, Changxi and Li, Ye and Wang, Ting},
  journal={Transportation research part C: emerging technologies},
  volume={154},
  pages={104263},
  year={2023},
  publisher={Elsevier}
}

@article{zou2024multi,
  title={Multi-task-based spatiotemporal generative inference network: A novel framework for predicting the highway traffic speed},
  author={Zou, Guojian and Lai, Ziliang and Wang, Ting and Liu, Zongshi and Bao, Jingjue and Ma, Changxi and Li, Ye and Fan, Jing},
  journal={Expert Systems with Applications},
  volume={237},
  pages={121548},
  year={2024},
  publisher={Elsevier}
}
```