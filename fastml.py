from fastapi import FastAPI
from pydantic import BaseModel
from xgboost import XGBClassifier, XGBRegressor
import pandas as pd
import pickle


app = FastAPI()

class ScoringItem(BaseModel):
    p1_class: str
    p1_hp: int
    p1_ac: int
    p1_avg_save: int
    p2_class: str
    p2_hp: int
    p2_ac: int
    p2_avg_save: int
    p3_class: str
    p3_hp: int
    p3_ac: int
    p3_avg_save: int
    p4_class: str
    p4_hp: int
    p4_ac: int
    p4_avg_save: int
    num_of_monsters: int
    monster_name: str
    monster_cr: int
    monster_ac: int
    monster_hp: int
    monster_type: str
    dificulty: int
    players_level: int

def encode_and_normalize_Data(data):
    with open('models\encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
        
    with open('models\\normalizer.pkl', 'rb') as f:
        normalizer = pickle.load(f)
        
    data_features_df = encoder.transform(data[['p1_class', 'p2_class', 'p3_class', 'p4_class', 'monster_type']])
    try:
        data_encoded = pd.concat([data, data_features_df], axis=1).drop(columns=['p1_class', 'p2_class', 'p3_class', 'p4_class', 'monster_type', 'monster_name','dificulty'])
    except:
        data_encoded = pd.concat([data, data_features_df], axis=1).drop(columns=['p1_class', 'p2_class', 'p3_class', 'p4_class', 'monster_type'])
        
    data_encoded_normalized = normalizer.transform(data_encoded)
    return data_encoded_normalized    

def predict_difficulty(data):
    regression_model = XGBRegressor()
    regression_model.load_model('models\model_OPT_NORMALIZED.ubj')
    prediction = round(regression_model.predict(data)[0], 3)

    if prediction < 0:
        prediction = 0.0
    elif prediction > 1:
        prediction = 1.0

    return 0 if prediction<0 else (1 if prediction > 1 else round(float(prediction), 3))    

def predict_tpk(data):
    classification_model = XGBClassifier()
    classification_model.load_model('models\model_opt_normalized_classification.ubj')
    return int(classification_model.predict(data)[0])

@app.post("/")
def read_root(item: ScoringItem):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    df_encoded_normalized = encode_and_normalize_Data(df)
    return {
        "difficulty_score": predict_difficulty(df_encoded_normalized),
        "probabal_tpk": predict_tpk(df_encoded_normalized),
        }