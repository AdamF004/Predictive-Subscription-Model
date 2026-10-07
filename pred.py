from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import pickle
import pandas as pd


app = FastAPI()
models = joblib.load('best_model.pkl')

def predict_y(df):
    prediction = models.predict([df])
    return prediction[0]

class Data_C(BaseModel):
    age: int
    job: int
    marital: int
    education: int
    default: int
    housing: int
    loan: int
    contact: int 
    duration: float 
    campaign: int 
    pdays: int 
    previous: int
    poutcome: int 

@app.get("/")
def read_root():
       return {"ketik": "link/docs"}

@app.post('/predict')
def predict(df: Data_C):
    input_data = [df.age, df.job, df.marital, df.education, df.default, df.housing, df.loan, df.contact
                  ,df.duration, df.campaign, df.pdays, df.previous, df.poutcome]
    prediction = predict_y(input_data)
    return {'prediction': prediction}
