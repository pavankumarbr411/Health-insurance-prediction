# 🏥 Health Insurance Premium Prediction

This project predicts the **health insurance charges** a person might have to pay based on their personal and lifestyle information using Machine Learning.

---

## 📌 Objective

To build a machine learning model that can predict **insurance charges** based on features such as:
- Age
- BMI (Body Mass Index)
- Number of children
- Gender
- Smoking habits
- Region

---

## 🛠️ Technologies & Libraries Used

| Category | Tools/Technologies |
|---------|---------------------|
| Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Model Evaluation | MAE, RMSE, MSE |
| Pipeline & Preprocessing | ColumnTransformer, OneHotEncoder, StandardScaler, SimpleImputer |
| Platform | Google Colab |

---

## 📊 Exploratory Data Analysis (EDA)

Performed basic analysis and visualizations:
- Age, BMI, Children, Charges Distribution (using `histplot`)
- Correlation Heatmap of numerical features

> **Purpose**: To understand feature distributions and relationships before modeling.

---

## 🧹 Data Preprocessing

- **Numerical Columns**:
  - Scaled using `StandardScaler`
  - Missing values handled using `SimpleImputer`
  
- **Categorical Columns**:
  - Converted using `OneHotEncoder`
  - Missing values filled with most frequent category

Used `ColumnTransformer` and `Pipeline` for clean preprocessing and modeling.

---

## 🤖 Machine Learning Models Used

### 1. 🔵 Linear Regression
- Simple and interpretable baseline model

### 2. 🌲 Random Forest Regressor
- Ensemble model using multiple decision trees for better accuracy

### 3. 🚀 XGBoost Regressor
- Advanced boosting algorithm
- Used with GPU acceleration (`tree_method='hist', device='cuda'`)

---

## 📈 Evaluation Metrics

Used to compare model performances:

- **MAE**: Mean Absolute Error  
- **RMSE**: Root Mean Squared Error  
- **MSE**: Mean Squared Error

---

## 🎯 Final Prediction

A user can enter personal details like:
- Age, BMI, Children
- Gender, Smoker/Non-smoker
- Region

The model predicts the **expected insurance premium** using the trained XGBoost model.

Example:
- Enter your age: 35
- Enter your BMI: 26.5
- Enter number of children: 2
- Enter gender (male/female): female
- Do you smoke? (yes/no): no
- Enter region: southwest

**Predicted Premium: ₹ 7,895.32**
The dataset is a commonly used medical cost dataset containing fictional insurance data:
- Columns: `age`, `gender`, `bmi`, `children`, `smoker`, `region`, `charges`

You can find it in:  
`insurance.csv`
