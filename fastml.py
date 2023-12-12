from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# class ScoringItem(BaseModel):
#     item = {
        
#     }

@app.get("/")
def read_root():
    return {"Hello": "World"}