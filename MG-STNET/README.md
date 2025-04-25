# Traffic- accident-risk-estimation

# MG-STNET
MG-STNET: A Multi-Graph Spatial-Temporal Network for Traffic Accident Risk Forecasting

# Usage

train model on NYC:
```
python train.py --config config/nyc/NYC_Config.json --gpus 0
```
test model on NYC
```
 python train.py --config config/nyc/NYC_Config.json --gpus 0 --test
```

train model on Chicago:
```
python train.py --config config/chicago/Chicago_Config.json --gpus 0
```

test model on Chicago:
```
python train.py --config config/chicago/Chicago_Config.json --gpus 0 --test
```

# Traffic Accident Risk Prediction Model Evaluation

## Evaluation Metrics

Six metrics are utilized to evaluate the traffic accident risk prediction model's performance. 

### Regression Task Metric
**Root Mean Square Error (RMSE)**  
For the regression-based evaluation of predicted accident risks:  
$\mathrm{RMSE}=\sqrt{\frac{1}{D} \sum_{i=1}^{D}\left(Y_i-\hat{Y}_i\right)^2}$  
*A lower RMSE indicates more accurate predictions in high-risk regions.*

### Classification Task Metrics
When treating risk prediction as a classification problem:

1. ​**Recall**  
   Measures coverage of actual high-risk areas:  
   $\text{Recall}=\frac{1}{D} \sum_{i=1}^{D} \frac{S_i \cap R_i}{|R_i|}$  

2. ​**Mean Average Precision (MAP)**  
   Evaluzes ranking accuracy of predicted risks:  
   $\mathrm{MAP}=\frac{1}{D} \sum_{i=1}^{D} \frac{\sum_{j=1}^{|R_i|} \text{pre}(j) \times \text{rel}(j)}{|R_i|}$  
   *Higher Recall and MAP scores indicate better performance in identifying high-risk zones.*

# Configuration

The configuration file config.json contains three parts: Data, Training and Predict:


# About

If you find this repository useful in your research, please cite the following paper:
```

@article{zou2024mt,
  title={MT-STNet: A Novel Multi-Task Spatiotemporal Network for Highway Traffic Flow Prediction},
  author={Zou, Guojian and Lai, Ziliang and Wang, Ting and Liu, Zongshi and Li, Ye},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={25},
  number={7},
  pages={8221--8236},
  year={2024},
  publisher={IEEE}
}


@article{zou2023will,
  title={When will we arrive? A novel multi-task spatio-temporal attention network based on individual preference for estimating travel time},
  author={Zou, Guojian and Lai, Ziliang and Ma, Changxi and Tu, Meiting and Fan, Jing and Li, Ye},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={24},
  number={10},
  pages={11438--11452},
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
