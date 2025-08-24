# 🌾 ML-Based Crop Recommender System

*Intelligent agricultural decision-making through machine learning*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-latest-green.svg)](https://xgboost.readthedocs.io/)

## 🎯 Overview

This project presents an intelligent crop recommendation system that leverages supervised machine learning to optimize agricultural decision-making. By analyzing soil properties and environmental conditions, the system provides data-driven crop suggestions to maximize yield potential and support sustainable farming practices.

## ✨ Key Features

- **🤖 Advanced ML Pipeline**: Comprehensive comparison of classical and state-of-the-art algorithms including neural networks, XGBoost, and ensemble methods
- **📊 Intelligent Data Processing**: Robust EDA, preprocessing, and feature engineering pipeline
- **🎛️ Optimized Performance**: Hyperparameter tuning using GridSearchCV for maximum predictive accuracy
- **🔬 Ensemble Intelligence**: Custom ensemble models that combine multiple algorithms for enhanced prediction reliability
- **📈 Actionable Insights**: Data-driven recommendations for yield optimization strategies

## 🛠️ Technical Stack

- **Machine Learning**: scikit-learn, XGBoost, TensorFlow/Keras
- **Data Analysis**: pandas, NumPy
- **Visualization**: matplotlib, seaborn
- **Model Optimization**: GridSearchCV, cross-validation
- **Dataset**: Kaggle Agricultural Dataset (soil & environmental features)

## 📊 Dataset Details

The dataset contains the following features used for crop classification:

**Data fields**
* `N` - ratio of Nitrogen content in soil
* `P` - ratio of Phosphorous content in soil
* `K` - ratio of Potassium content in soil
* `temperature` - temperature in degree Celsius
* `humidity` - relative humidity in %
* `ph` - ph value of the soil
* `rainfall` - rainfall in mm
* `label` - crop name

This is a **classification problem** where I used all the soil and environmental features (N, P, K, temperature, humidity, pH, rainfall) to predict the appropriate crop name using various machine learning models.

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
Anaconda Distribution
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ML-Based-Crop-Recommender-System.git
   cd ML-Based-Crop-Recommender-System
   ```

2. **Install Anaconda**
   - Download Anaconda from [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
   - Follow the installation instructions for your operating system
   - Create a new conda environment:
   ```bash
   conda create -n crop-recommender python=3.8
   conda activate crop-recommender
   ```
   - Install required packages:
   ```bash
   conda install pandas numpy scikit-learn matplotlib seaborn jupyter
   pip install xgboost
   ```

3. **Download the dataset**
   - Download `Crop_recommendation.csv` from Kaggle
   - Place it in the project directory

### Quick Start

You can explore the project through two main notebooks:

- **`Crop Prediction (80-20).ipynb`**: Implementation with 80-20 train-test split
- **`Crop prediction project.ipynb`**: Implementation with 70-30 train-test split

Download both notebooks to compare and analyze the effect of different train-test split ratios on model performance.

## 📊 Model Performance

My comprehensive model comparison achieved excellent performance across all algorithms:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Decision Tree | 98.86% | 98.82% | 98.96% | 98.88% |
| Random Forest | 99.55% | 99.57% | 99.62% | 99.58% |
| Logistic Regression | 96.59% | 96.53% | 96.70% | 96.57% |
| Support Vector Machine | 98.18% | 98.04% | 98.43% | 98.15% |
| K-Nearest Neighbour | 97.05% | 97.01% | 97.33% | 97.00% |
| Categorical Naive Bayes | 98.64% | 98.60% | 98.41% | 98.43% |
| Gaussian Naive Bayes | 98.86% | 98.84% | 98.73% | 98.77% |
| XGBoost | 99.32% | 99.32% | 99.42% | 99.34% |
| Bagging Classifier | 99.09% | 99.11% | 98.95% | 99.02% |
| Voting Classifier | 99.55% | 99.57% | 99.62% | 99.58% |
| **Stacking Classifier** | **99.77%** | **99.77%** | **99.81%** | **99.79%** |

## 🔍 Key Insights

- **Environmental Factors**: Temperature and humidity showed strongest correlation with crop suitability
- **Soil Chemistry**: NPK ratios proved critical for accurate crop classification
- **Ensemble Benefits**: Combining multiple models reduced overfitting and improved generalization
- **Feature Importance**: Rainfall patterns and pH levels emerged as top predictive features

## 📈 Agricultural Impact

This system enables:
- **Yield Optimization**: Data-driven crop selection for maximum productivity
- **Resource Efficiency**: Optimal use of fertilizers and water resources
- **Risk Mitigation**: Environmental factor analysis for better planning
- **Sustainable Farming**: Evidence-based decisions supporting long-term soil health

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- Kaggle for providing the comprehensive agricultural dataset
- The open-source community for excellent ML libraries

## 📧 Contact

**Your Name** - manaskd2019@gmail.com

Project Link: [https://github.com/Manas120104/ML-Based-Crop-Recommender-System](https://github.com/Manas120104/ML-Based-Crop-Recommender-System)

---

*Empowering farmers with intelligent, data-driven crop recommendations* 🌱
