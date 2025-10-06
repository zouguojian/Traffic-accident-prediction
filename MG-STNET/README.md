# MG-STNET
MG-STNET: A Multi-Graph Spatial-Temporal Network for Traffic Accident Risk Forecasting

# Usage

Got it ✅ Here’s a clean, reproducible step-by-step guide you can include directly in your **README** or **supplementary materials**. I’ve kept the commands general but tailored for your MG-STNET repo:

---

## Step-by-Step Guide to Reproduce MG-STNET

### 1. Clone the Repository

```bash
git clone https://github.com/zouguojian/Traffic-accident-prediction.git
cd Traffic-accident-prediction/MG-STNET
```

### 2. Set Up the Python Environment

It is recommended to use **Python 3.8+** and create a virtual environment:

```bash
# Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate        # On Linux/Mac
venv\Scripts\activate           # On Windows
```

### 3. Install Dependencies

All required packages are listed in `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare the Dataset

* Download the **NYC** and **Chicago** traffic accident datasets (links provided in the paper or project page).
* Place the raw data files under the `./data/` directory, e.g.:

  ```
  data/
    ├── NYC/
    │     ├── all_data.pkl
    │     ├── grid_node_map.pkl
    │     └── ......
    └── Chicago/
          ├── all_data.pkl
          ├── grid_node_map.pkl
          └── ......
  ```
* **Dataset split process:** The dataset was divided into training, validation, and test sets in chronological order to preserve temporal causality, consistent with all baseline models. Within the training process, the training set was randomly shuffled at each epoch. We did not use random resampling splits, as these would break temporal ordering and cause information leakage.


### 5. Train the Model

Run the training script (example with NYC dataset):

train model on NYC:
```
python train.py --config config/nyc/NYC_Config.json --gpus 0
```

train model on Chicago:
```
python train.py --config config/chicago/Chicago_Config.json --gpus 0
```

### 6. Evaluate the Model

After training, evaluation results will be automatically generated.
Alternatively, you can run evaluation directly:

test model on NYC
```
 python train.py --config config/nyc/NYC_Config.json --gpus 0 --test
```

test model on Chicago:
```
python train.py --config config/chicago/Chicago_Config.json --gpus 0 --test
```

### 7. Results and Logs

* Training logs are stored in `./logs/`
* Model checkpoints are saved in `./checkpoints/`
* Final performance tables (RMSE, Recall, MAP) are reported in the console and log files.



# Traffic Accident Risk Prediction Model Evaluation

## Evaluation Metrics

Six metrics are utilized to evaluate the traffic accident risk prediction model's performance. 

### Regression Task Metric
1. **Root Mean Square Error (RMSE)**  
   For the regression-based evaluation of predicted accident risks:  

   $\mathrm{RMSE}=\sqrt{\frac{1}{D} \sum_{i=1}^{D}\left(Y_i-\hat{Y}_i\right)^2}$

   *A lower RMSE indicates more accurate predictions in high-risk regions.*

### Classification Task Metrics
When treating risk prediction as a classification problem:

2. ​**Recall**  
   Measures coverage of actual high-risk areas:  
   
   $\text{Recall}=\frac{1}{D} \sum_{i=1}^{D} \frac{S_i \cap R_i}{|R_i|}$  

3. ​**Mean Average Precision (MAP)**  
   Evaluzes ranking accuracy of predicted risks:
    
   $\mathrm{MAP}=\frac{1}{D} \sum_{i=1}^{D} \frac{\sum_{j=1}^{|R_i|} \text{pre}(j) \times \text{rel}(j)}{|R_i|}$
   
   *Higher Recall and MAP scores indicate better performance in identifying high-risk zones.*


## 🧩 Performance Reproducibility and Confidence Intervals

To enhance the **transparency** and **reproducibility** of our experimental results, we conducted additional evaluations across multiple random seeds and computed statistical uncertainty estimates for all reproducible models.

### 🔁 Experimental Setup

* **Datasets:** New York (NYC) and Chicago urban accident datasets
* **Models evaluated:** MG-STNET, GSNet, C-ViT, TWCCnet, and MGHSTN (publicly available implementations)
* **Evaluation metrics:** RMSE, Recall, and MAP
* **Number of runs:** 5 independent runs per model
* **Random seeds:** {100, 500, 1000, 1500, 2019}
* **Environment consistency:** All experiments were conducted using identical data splits, hyperparameter settings, and evaluation protocols to ensure fair comparison.
* **Seed initialization:** Followed the same random seed configuration as GSNet
  ([https://github.com/Echohhhhhh/GSNet/tree/master/config](https://github.com/Echohhhhhh/GSNet/tree/master/config))

### 📊 Reported Statistics

For each model and dataset, we report:

| Metric     | Description             | Reported as                    |
| :--------- | :---------------------- | :----------------------------- |
| **RMSE**   | Root Mean Squared Error | Mean ± 95% Confidence Interval |
| **Recall** | Detection sensitivity   | Mean ± 95% Confidence Interval |
| **MAP**    | Mean Average Precision  | Mean ± 95% Confidence Interval |

Confidence intervals were estimated using **bootstrap resampling** over the five independent runs (sample size = 1000).

### 📁 Files

All results are located in the [`CIs`](https://github.com/zouguojian/Traffic-accident-prediction/blob/main/MG-STNET/Reporting%20confidence%20intervals%20for%20RMSE%2C%20Recall%2C%20and%20MAP.docx).

Each file contains the following columns:

| Dataset | Model | Seed | RMSE | Recall | MAP |
| :---- | :----- | :--- | :-- | :------------- | :------------- |

### 🧠 Notes

* Some baseline models (e.g., SDCAE, Hetero-ConvLSTM) are **not open-sourced**, and thus confidence intervals cannot be computed consistently across all methods.
* All results reported in the main paper (Tables R1 and R2) correspond to the **mean performance** under consistent conditions.
* Readers are encouraged to reproduce the experiments.

This repository is designed to facilitate **fully transparent, repeatable experimentation** for traffic accident risk prediction research.


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
