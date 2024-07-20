from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
from scipy.stats import ks_2samp
import logging

app = FastAPI()
logging.basicConfig(level=logging.INFO)

class ScoringItem(BaseModel):
    gender: str
    is_employed: int
    job: str
    location: str
    loan_amount: float
    number_of_defaults: int
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
        return -1  # Handle unexpected values

    @staticmethod
    def recoded_status(marital_status):
        if marital_status == 'single':
            return 0
        elif marital_status == 'married':
            return 1
        elif marital_status == 'divorced':
            return 2
        return -1  # Handle unexpected values

# Load the model
with open('decisiontree.sav', 'rb') as f:
    model = pickle.load(f)

# Load baseline distributions
with open('baseline_distributions.pkl', 'rb') as f:
    baseline_distributions = pickle.load(f)

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

def encode_location(city_name):
    return location_mapping.get(city_name, -1)  # Return -1 if city_name not found

def encode_job(job_title):
    return job_mapping.get(job_title, -1)  # Return -1 if job_title not found

def check_data_drift(new_data, baseline_distributions, threshold=0.05):
    drift_detected = False
    drift_details = []
    for column in new_data.columns:
        if column in baseline_distributions:
            baseline = baseline_distributions[column]
            if isinstance(baseline, dict) and 'mean' in baseline:
                mean_baseline = baseline['mean']
                std_baseline = baseline['std']
                mean_new = new_data[column].mean()
                std_new = new_data[column].std() if new_data[column].std() != 0 else 1
                z_score_mean = np.abs(mean_baseline - mean_new) / std_baseline
                if z_score_mean > threshold:
                    drift_detected = True
                    drift_details.append(f"Continuous feature '{column}' shows drift with Z-score: {z_score_mean}")

            elif isinstance(baseline, dict):
                freq_baseline = baseline
                freq_new = new_data[column].value_counts().to_dict()
                for category, count in freq_new.items():
                    baseline_prob = freq_baseline.get(category, 0) / sum(freq_baseline.values())
                    new_prob = count / len(new_data)
                    if np.abs(baseline_prob - new_prob) > threshold:
                        drift_detected = True
                        drift_details.append(f"Categorical feature '{column}' shows drift for category '{category}'")

                if len(freq_new) > len(freq_baseline):
                    drift_detected = True
                    drift_details.append(f"Categorical feature '{column}' has new categories")

    return drift_detected, drift_details

@app.post('/predict')
async def scoring_endpoint(item: ScoringItem):
    # Encode categorical variables
    item_dict = item.dict()
    item_dict['location'] = encode_location(item.location)
    item_dict['job'] = encode_job(item.job)
    item_dict['gender'] = ScoringItem.recoded_gender(item.gender)
    item_dict['marital_status'] = ScoringItem.recoded_status(item.marital_status)
    item_dict['remaining term'] = item_dict.pop('remaining_term')  # Adjust key to match column name

    # Define the correct feature order
    feature_order = [
        'gender', 'is_employed', 'job', 'location', 'loan_amount',
        'number_of_defaults', 'outstanding_balance', 'interest_rate', 'age',
        'remaining term', 'salary', 'marital_status'
    ]

    # Convert item to DataFrame with correct column order
    df = pd.DataFrame([item_dict], columns=feature_order)
    
    # Check for data drift
    drift_detected, drift_details = check_data_drift(df, baseline_distributions)
    
    if drift_detected:
        logging.info("Data drift detected:")
        for detail in drift_details:
            logging.info(detail)
        # Return a prediction with a drift warning
        prediction = model.predict(df)
        return {'prediction': int(prediction[0]), 'drift_warning': drift_details}

    # Predict using the loaded model
    yhat = model.predict(df)
    prediction = int(yhat[0])
    
    # Return the prediction
    return {'prediction': prediction}
