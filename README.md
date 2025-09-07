# 🌦️ AI-Enhanced Weather Forecasting  

## 📝 Project Overview  
This project aims to predict **whether it will rain tomorrow** using machine learning techniques on the Australian weather dataset.  
It covers the full data science pipeline — from **data cleaning, exploratory data analysis (EDA), feature engineering, model training, evaluation, and deployment** into a live Streamlit app.  

By combining statistical insights and modern ML models, this project answers questions like:  
- *Can tomorrow’s rainfall be predicted reliably from today’s weather?*  
- *Which weather factors (temperature, humidity, pressure, etc.) are most influential?*  
- *Which model performs best for this classification task?*  

---

## 📑 Table of Contents  
- [📂 Repo Structure](#-repo-structure)  
- [🎯 Project Goals](#-project-goals)  
- [📊 Data](#-data)  
- [⚡ Quick Start](#-quick-start)  
- [🧹 Data Cleaning](#-data-cleaning)  
- [🔎 Exploratory Data Analysis](#-exploratory-data-analysis)  
- [🤖 Modeling & Evaluation](#-modeling--evaluation)  
- [🚀 Deployment](#-deployment)  
- [📊 Visualizations](#-visualizations)  
- [📌 Results](#-results)  
- [🌐 Live Demo](#-live-demo)  
- [📜 License](#-license)  

---

## 📂 Repo Structure  
```
AI-Enhanced-Weather-Forecating/
├─ Data Cleaning/                 # Scripts for preprocessing & cleaning
├─ EDA/                           # Exploratory Data Analysis notebooks
├─ Modeling/                      # Model training scripts
├─ Model Evaluation/              # Comparison & evaluation results
├─ Deployment/                    # Streamlit app code
├─ Report/                        # Final project report (PDF)
├─ Images/                        # Visualizations & plots
├─ weatherAUS.csv                 # Raw dataset
└─ README.md                      # Project readme
```

---

## 🎯 Project Goals  
- Build a predictive model to determine if it will **rain tomorrow**.  
- Explore relationships between weather features (temperature, humidity, pressure, etc.).  
- Compare multiple models: **Logistic Regression, Random Forest, and XGBoost**.  
- Deploy a **user-friendly app** for real-time predictions.  

---

## 📊 Data  
- **Dataset:** `weatherAUS.csv`  
- **Source:** Kaggle
- **Size:** ~142,000 rows × 30 columns  

---

## ⚡ Quick Start  

Clone the repo and set up the environment:  
```bash
# Clone repository
git clone https://github.com/Sohitha-01/AI-Enhanced-Weather-Forecating.git
cd AI-Enhanced-Weather-Forecating

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
cd Deployment
streamlit run app.py
```

---

## 🧹 Data Cleaning  
- Handled **missing values** through imputation.  
- Encoded **categorical features** (wind direction, location).  
- Engineered **lag features** like `Rainfall_lag1`, `Humidity3pm_lag1`.  
- Scaled and standardized features for ML.  

---

## 🔎 Exploratory Data Analysis  
Key insights from EDA:  
- The dataset is **imbalanced**: only ~22% of days had rain tomorrow.  
- Strong correlations exist between **humidity, rainfall, and pressure**.  
- Seasonal effects observed: rain is more likely in mid-year months.  

📊 Example Plots (all images in repo):  

[Correlation Heatmap](https://github.com/Sohitha-01/AI-Enhanced-Weather-Forecating/blob/4e67bb559c26cc1760e5ed444b3e1adad0b1a292/Images/Correlation%20HeatMap.png)                           [Feature Importance](https://github.com/Sohitha-01/AI-Enhanced-Weather-Forecating/blob/4e67bb559c26cc1760e5ed444b3e1adad0b1a292/Images/Feature%20Importance.png)                         [Model Comparison](https://github.com/Sohitha-01/AI-Enhanced-Weather-Forecating/blob/4e67bb559c26cc1760e5ed444b3e1adad0b1a292/Images/Model%20Comparison.png)                              [Pairwise Views](https://github.com/Sohitha-01/AI-Enhanced-Weather-Forecating/blob/4e67bb559c26cc1760e5ed444b3e1adad0b1a292/Images/Pairwise%20views.png)                                            [Rain Probability](https://github.com/Sohitha-01/AI-Enhanced-Weather-Forecating/blob/4e67bb559c26cc1760e5ed444b3e1adad0b1a292/Images/Rain%20Probability.png)                                                [Target Distribution](https://github.com/Sohitha-01/AI-Enhanced-Weather-Forecating/blob/4e67bb559c26cc1760e5ed444b3e1adad0b1a292/Images/Target%20Distribution.png)



---

## 🤖 Modeling & Evaluation  
We trained and compared:  
- **Logistic Regression**  
- **Random Forest**  
- **XGBoost**  

Metrics used: **Accuracy, Precision, Recall, ROC-AUC**.  

📊 Model Comparison:  
| Model              | ROC-AUC | Accuracy | Precision | Recall |
|--------------------|---------|----------|-----------|--------|
| Logistic Regression| 0.88    | 83%      | 61%       | 68%    |
| Random Forest      | 0.89    | 84%      | 64%       | 69%    |
| XGBoost            | **0.91**| **85%**  | **65%**   | **71%**|  

✅ **XGBoost was selected as the final model.**

---

## 🚀 Deployment  
The project was deployed as a **Streamlit web app**.  

Steps to run locally:  
```bash
cd Deployment
streamlit run app.py
```

Deployed version hosted on **Streamlit Cloud** for public access.  

---

## 📊 Visualizations  
A few key plots (see `Images/` for all):  

- **Target Distribution**: Distribution of the target variable `RainTomorrow`. Only ~22% of days had rain the next day. 
- **Correlation Heatmap**: Highlights strongest relationships (humidity, pressure).  
- **Rain Probability by Month**: Peaks in June–July, dips in October.  
- **Feature Importance**: Highlights the most important predictors for rainfall. 
- **Model Comparison**: XGBoost outperforms others.  

---

## 📌 Results  
- 🌧️ Rain prediction achieved with **85% accuracy and 0.91 ROC-AUC**.  
- 🌡️ **Humidity at 3PM** was the most influential feature.  
- 📊 Seasonal & geographic patterns strongly affect rainfall.  
- 🚀 Successfully deployed a **live interactive app** for predictions.

📄 **Full Report (PDF):** [AI_Enhanced_Weather_Forecasting.pdf](https://github.com/Sohitha-01/AI-Enhanced-Weather-Forecating/blob/c20508bb0b4c36e308f91377518da3dae3512f0b/Report/AI_Enhanced_Weather_Forecasting.pdf)  

---

## 🌐 Live Demo  
Try the app here: [Rain Tomorrow Predictor](https://rain-tomorrow-predictor.streamlit.app/)  

---

## 📜 License  
This project is open-source under the MIT License.  
