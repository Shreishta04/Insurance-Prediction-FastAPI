from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pandas as pd
import pickle
from fastapi.middleware.cors import CORSMiddleware

model = None

@asynccontextmanager
async def lifespan(app:FastAPI):
    global model
    with open("model/insurance_model.pkl","rb") as f:
        model = pickle.load(f)
    print("Model Loaded Successfully!!")
    yield
    print("App Shutting Down!")

app = FastAPI(
    title = "Application for Insurance Prediction",
    lifespan = lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class UserInput(BaseModel):
    age : int
    body_part : str
    nature_of_injury : str
    claim_type : str
    medical_amount : float

@app.get("/")
def home():
    return {"status": "Insurance model API is running"}


@app.post("/predict")
def predict(data: UserInput):
    input_df = pd.DataFrame([data.model_dump()])
    input_df["body_part"] = input_df["body_part"].str.strip().str.title()
    input_df["nature_of_injury"] = input_df["nature_of_injury"].str.strip().str.title()
    input_df["claim_type"] = input_df["claim_type"].str.strip().str.title()
    
    prediction = model.predict(input_df)
    return {
        "estimated_insurance_amount": float(prediction[0])
    }

