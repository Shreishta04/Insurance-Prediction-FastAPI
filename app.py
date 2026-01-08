from dotenv import load_dotenv
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import pandas as pd
import pickle
import os

from database import engine
from models import Base

from database import SessionLocal
from models import Prediction

model = None

load_dotenv()

# Create DB tables if they don't exist
Base.metadata.create_all(bind=engine)

# Load ML model once
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    with open("model/insurance_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ Model Loaded Successfully")
    yield
    print("❌ App Shutting Down")

app = FastAPI(
    title="Application for Insurance Prediction",
    lifespan=lifespan
)

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_ORIGIN,
        "http://localhost:3000",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DB session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Input schema
class UserInput(BaseModel):
    age: int
    body_part: str
    nature_of_injury: str
    claim_type: str
    medical_amount: float

@app.get("/")
def home():
    return {"status": "Insurance model API is running"}

@app.post("/predict")
def predict(data: UserInput, db: Session = Depends(get_db)):
    input_df = pd.DataFrame([data.model_dump()])

    # Clean input
    input_df["body_part"] = input_df["body_part"].str.strip().str.title()
    input_df["nature_of_injury"] = input_df["nature_of_injury"].str.strip().str.title()
    input_df["claim_type"] = input_df["claim_type"].str.strip().str.title()

    # Prediction
    prediction = model.predict(input_df)[0]

    # Save to DB
    record = Prediction(
        age=data.age,
        body_part=input_df["body_part"][0],
        nature_of_injury=input_df["nature_of_injury"][0],
        claim_type=input_df["claim_type"][0],
        medical_amount=data.medical_amount,
        predicted_amount=float(prediction)
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "estimated_insurance_amount": float(prediction)
    }
