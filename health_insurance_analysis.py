from google.colab import drive
drive.mount('/content/drive')
!ls "/content/drive/MyDrive"


!pip install -q gradio xgboost scikit-learn pandas

import pandas as pd
import gradio as gr
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -----------------------------------------------------------
# 1. Load dataset directly (NO upload)
# -----------------------------------------------------------

# CHANGE THIS PATH IF YOUR FILE IS IN GOOGLE DRIVE
data = pd.read_csv("/content/drive/MyDrive/insurance.csv")   # <= keep CSV in same folder
print("Loaded successfully!")
print(data.head())

data.columns = data.columns.str.lower().str.strip()

# Prepare data
X = data.drop("charges", axis=1)
y = data["charges"]

categorical = ["sex", "smoker", "region"]
numerical = ["age", "bmi", "children"]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), numerical),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

X_proc = preprocess.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_proc, y, test_size=0.2, random_state=42
)

# Train model once
model = XGBRegressor(objective="reg:squarederror", random_state=42)
model.fit(X_train, y_train)

# -----------------------------------------------------------
# 2. Prediction function
# -----------------------------------------------------------
def predict(age, bmi, children, sex, smoker, region):

    df = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "sex": [sex],
        "smoker": [smoker],
        "region": [region]
    })

    df["sex"] = df["sex"].str.lower()
    df["smoker"] = df["smoker"].str.lower()
    df["region"] = df["region"].str.lower()

    x_input = preprocess.transform(df)
    pred = model.predict(x_input)[0]

    return f"Predicted Insurance Premium: ₹ {round(pred, 2)}"


# -----------------------------------------------------------
# 3. Simple plain Gradio UI (NO gradient, NO upload)
# -----------------------------------------------------------
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age", value=25),
        gr.Number(label="BMI", value=28),
        gr.Number(label="Children", value=0),
        gr.Dropdown(["male", "female"], label="Sex"),
        gr.Dropdown(["yes", "no"], label="Smoker"),
        gr.Dropdown(["northeast","northwest","southeast","southwest"], label="Region"),
    ],
    outputs=gr.Textbox(label="Premium"),
    title="Simple Health Insurance Premium Predictor",
    description="Uses insurance.csv loaded directly — no upload required."
)

interface.launch()
