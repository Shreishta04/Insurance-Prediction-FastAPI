import pandas as pd
import pickle

with open("model/insurance_model.pkl", "rb") as f:
    model = pickle.load(f)

test_input = pd.DataFrame([{
    "age": 40,
    "body_part": "Leg",
    "nature_of_injury": "Fracture",
    "claim_type": "Accident",
    "medical_amount": 35000
}])

print(model.predict(test_input))
