import os
from dotenv import load_dotenv

load_dotenv()

# Get project root directory (parent of backend)
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

ENV = os.getenv("ENV", "dev")
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(PROJECT_ROOT, "data", "models"))
MODEL_FILE = os.getenv("MODEL_FILE", "model.joblib")
SHAP_BACKGROUND_FILE = os.getenv("SHAP_BACKGROUND_FILE", "background.npy")
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
