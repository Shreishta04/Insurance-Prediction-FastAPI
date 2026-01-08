import pandas as pd
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

data = {
    "age": [25, 42, 30, 55, 38, 46, 29],
    "body_part": [
        "Arm", "Leg", "Head", "Back", "Arm", "Leg", "Chest"
    ],
    "nature_of_injury": [
        "Fracture", "Sprain", "Burn", "Fracture", "Cut", "Fracture", "Burn"
    ],
    "claim_type": [
        "Accident", "Surgery", "Accident", "Hospitalization",
        "Surgery", "Accident", "Hospitalization"
    ],
    "medical_amount": [
        18000, 32000, 25000, 45000, 28000, 38000, 41000
    ],
    # ✅ TARGET (what we want to predict)
    "insurance_amount": [
        15000, 26000, 20000, 36000, 23000, 30000, 33000
    ]
}

df = pd.DataFrame(data)

X = df.drop("insurance_amount", axis=1)
y = df["insurance_amount"]


categorical_features = [
    "body_part",
    "nature_of_injury",
    "claim_type"
]

numeric_features = [
    "age",
    "medical_amount"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

model_pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("regressor", LinearRegression())
])

model_pipeline.fit(X, y)

with open("model/insurance_model.pkl", "wb") as f:
    pickle.dump(model_pipeline, f)

print("Insurance prediction model trained and saved successfully!")
