from sqlalchemy import Column, Integer, String, Float
from database import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer)
    body_part = Column(String)
    nature_of_injury = Column(String)
    claim_type = Column(String)
    medical_amount = Column(Float)
    predicted_amount = Column(Float)
