"""
Configuration management for the Agentic ASL Conversation Assistant.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables if .env exists
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "saved_models"
DATASET_VIDEO_DIR = Path(os.getenv("ASL_CITIZEN_VIDEOS", r"E:\ASL_Citizen\ASL_Citizen\videos"))
EXTRACTED_LANDMARKS_DIR = BASE_DIR / "extracted_landmarks"

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Model parameters
SEQUENCE_LENGTH = 30
FEATURE_DIM = 144  # 126 hand landmarks (21 left * 3 + 21 right * 3) + 18 body-relative anchor features
NUM_CLASSES = 59   # Full vocabulary classes (39 Core + 20 Extension)

# Server parameters
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# Agent timeouts (seconds)
TIMEOUT_VISION = 0.5
TIMEOUT_SIGN = 0.5
TIMEOUT_CONTEXT = 0.5
TIMEOUT_LANGUAGE = 2.0
TIMEOUT_SPEECH = 1.0
TIMEOUT_MEMORY = 0.2
