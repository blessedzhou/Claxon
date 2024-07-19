from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI()

class ScoringItem(BaseModel):
    gender: str
    is_employed: int
    job: str
    location: str
    loan_amount: float
    Number_of_defaults: int
    outstanding_balance: float
    interest_rate: float
    age: int
    remaining_term: float
    salary: float
    marital_status: str

    @staticmethod
    def recoded_gender(gender):
        if gender == 'female':
            return 0
        elif gender == 'male':
            return 1
        elif gender == 'other':
            return 2

    @staticmethod
    def recoded_status(marital_status):
        if marital_status == 'single':
            return 0
        elif marital_status == 'married':
            return 1
        elif marital_status == 'divorced':
            return 2

# Load the model
with open('decisiontree.sav', 'rb') as f:
    model = pickle.load(f)

# Pre-defined mappings
location_mapping = {
    'Beitbridge': 0, 'Harare': 8, 'Gweru': 7, 'Rusape': 20, 'Chipinge': 3,
    'Chimanimani': 2, 'Marondera': 14, 'Kadoma': 10, 'Mutare': 16, 'Masvingo': 15,
    'Bulawayo': 1, 'Kariba': 11, 'Plumtree': 18, 'Chiredzi': 4, 'Shurugwi': 21,
    'Chivhu': 5, 'Zvishavane': 23, 'Nyanga': 17, 'Karoi': 12, 'Redcliff': 19,
    'Kwekwe': 13, 'Gokwe': 6, 'Victoria falls': 22, 'Hwange': 9
}

job_mapping = {
    'Teacher': 8, 'Nurse': 6, 'Doctor': 3, 'Data analyst': 1,
    'Software developer': 7, 'Accountant': 0, 'Lawyer': 5, 'Engineer': 4,
    'Data scientist': 2
}

# Function to map city name to encoded label
def encode_location(city_name):
    return location_mapping.get(city_name, -1)  # Return -1 if city_name not found (handle error as needed)

# Function to map job title to encoded label
def encode_job(job_title):
    return job_mapping.get(job_title, -1)  # Return -1 if job_title not found (handle error as needed)

@app.get('/')
async def scoring_endpoint(item: ScoringItem):
    # Encode categorical variables
    item.location = encode_location(item.location)
    item.job = encode_job(item.job)
    # Convert item to DataFrame
    df = pd.DataFrame([item.dict()])
    # Predict using the loaded model
    yhat = model.predict(df)
    # Return the prediction
    return {'prediction': yhat[0]}
