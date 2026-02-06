"""
Configuration settings for the Sentiment Analysis Application.
This module contains all configuration parameters used throughout the application.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Model Configuration
MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "distilbert-base-uncased-finetuned-sst-2-english"
)
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./models")

# Alternative 3-class sentiment model (positive, negative, neutral)
# Uncomment to use 3-class classification instead of binary
# MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

# Application Configuration
MAX_TEXT_LENGTH = 512  # Maximum number of tokens for transformer models
BATCH_SIZE = 32
CONFIDENCE_THRESHOLD = 0.60  # Threshold for neutral classification in binary models

# File Upload Configuration
MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = ['.txt']

# UI Configuration
PAGE_TITLE = "Sentiment Analysis Application"
PAGE_ICON = ":bar_chart:"

# Visualization Colors
COLOR_POSITIVE = "#28a745"  # Green
COLOR_NEGATIVE = "#dc3545"  # Red
COLOR_NEUTRAL = "#6c757d"   # Gray

# NLTK Data
NLTK_DATA_PACKAGES = [
    'punkt',
    'stopwords',
    'wordnet',
    'averaged_perceptron_tagger',
    'omw-1.4'
]
