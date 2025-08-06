from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

from src.data_preprocessing import create_new_features

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

from pydantic import BaseModel

class InputData(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: float
    NumberOfTime30_59DaysPastDueNotWorse: float
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: float
    NumberOfTimes90DaysLate: float
    NumberRealEstateLoansOrLines: float
    NumberOfTime60_89DaysPastDueNotWorse: float
    NumberOfDependents: float
        
import os
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = joblib.load(model_path)
        
@app.post("/predict")
def predict_score(data: InputData):
    raw_df = pd.DataFrame([data.dict()])
    
    engineered_df = create_new_features(raw_df)
    
    prediction = model.predict(engineered_df)[0]
    
    return {"prediction": float(prediction)}