import os
from dotenv import load_dotenv

load_dotenv()

ENV = os.getenv("ENV", "dev")
MODEL_DIR = os.getenv("MODEL_DIR", "data/models")
MODEL_FILE = os.getenv("MODEL_FILE", "model.joblib")
SHAP_BACKGROUND_FILE = os.getenv("SHAP_BACKGROUND_FILE", "background.npy")
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "http://localhost:3000").split(",")
