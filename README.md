# ml-soil-nitrogen-prediction

# 🌱 Predicting Soil Nitrogen Using Supervised Machine Learning

This project demonstrates how to apply **regression machine learning algorithms** to predict **soil nitrogen levels** based on environmental and crop-related data. It is built to highlight **ML best practices**, **model comparison**, and **hyperparameter tuning**, making it ideal for showcasing ML expertise.

---

## 📌 Problem Statement

Soil nitrogen is a critical nutrient that supports plant growth and affects agricultural productivity. Traditional chemical testing is costly and time-consuming. This project uses **machine learning** to predict nitrogen levels from easily measurable features such as:

- Temperature
- Humidity
- Soil moisture
- Soil type
- Crop type
- Phosphorous and Potassium content

---

## 📊 Dataset

- **Source:** Provided manually (Kaggle-style structure)
- **Rows:** 8000 samples  
- **Features:**  
  - `Temparature`, `Humidity`, `Moisture` (numeric)
  - `Soil Type`, `Crop Type` (categorical)
  - `Phosphorous`, `Potassium` (numeric)
- **Target variable:** `Nitrogen` (numeric)

---

## 🔍 Machine Learning Approach

### Step 1: Preprocessing
- **Numerical features** were scaled using `StandardScaler`
- **Categorical features** (`Soil Type`, `Crop Type`) were one-hot encoded using `OneHotEncoder`

### Step 2: Models Compared
We evaluated six different regression models:

| Model             | Type                  |
|------------------|-----------------------|
| Linear Regression| Baseline linear model |
| Ridge            | Linear + L2 regularization |
| Lasso            | Linear + L1 regularization |
| Random Forest    | Tree ensemble model   |
| XGBoost          | Gradient boosting     |
| LightGBM         | Fast gradient boosting|

### Step 3: Train/Test Split
We initially used an **80/20 split** (80% training, 20% test) to simulate real-world generalization. Later, we tested 70/30 and 90/10 as well.

---

## ⚙️ Hyperparameter Tuning

We used **`GridSearchCV`** to tune the `RandomForestRegressor`:
```
```python
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5]
}```

- **Best parameters found:**
  - `n_estimators = 200`
  - `max_depth = 20`
  - `min_samples_split = 2`

---

## 📈 Results

### 🔬 Model Comparison (80/20 split)
```
| Model           | R² Score | RMSE    | MAE     |
|----------------|----------|---------|---------|
| Random Forest  | 0.8600   | 4.41    | 3.30    |
| LightGBM       | 0.8579   | 4.45    | 3.30    |
| XGBoost        | 0.8498   | 4.57    | 3.43    |
| Ridge          | 0.5505   | 7.91    | 6.16    |
| Linear Reg.    | 0.5505   | 7.91    | 6.16    |
| Lasso          | 0.5342   | 8.05    | 6.37    |

✅ **Random Forest** and **LightGBM** were the top-performing models.

---

### 🧪 Performance by Train/Test Split

| Train:Test | R² Score | RMSE   | MAE   |
|------------|----------|--------|--------|
| 90:10      | 0.8688   | 4.28   | 3.29   |
| 80:20      | 0.8600   | 4.41   | 3.30   |
| 70:30      | 0.8670   | 4.30   | 3.27   |

💡 The model remains **stable across data splits**, showing robustness.

---

## 📉 Visualization

The predicted nitrogen values vs. actual values were plotted to assess regression fit. Most points are close to the ideal diagonal line, confirming strong predictive accuracy.

---

## 📦 Technologies Used

- Python 3.10
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- xgboost
- lightgbm

---

## 📚 What You’ll Learn from This Project

- How to preprocess mixed tabular data
- Model comparison for regression
- Hyperparameter tuning with `GridSearchCV`
- Evaluation using RMSE, MAE, and R²
- Impact of different train-test splits

---

## 🧠 Author Notes

This project was created by **Konlavach Mengsuwan** as a way to showcase hands-on machine learning expertise with tabular environmental/agricultural data. It's intended for portfolios, job applications, and educational sharing.

---


