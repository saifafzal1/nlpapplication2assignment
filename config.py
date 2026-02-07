"""
Configuration settings for the Sentiment Analysis Application.
This module contains all configuration parameters used throughout the application.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Model Configuration
# Using Twitter RoBERTa fine-tuned for 3-class sentiment (NEGATIVE, NEUTRAL, POSITIVE)
# This model natively supports neutral sentiment classification
# Previous model: distilbert-base-uncased-finetuned-sst-2-english (binary only)
MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "cardiffnlp/twitter-roberta-base-sentiment"
)
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./models")

# Application Configuration
MAX_TEXT_LENGTH = 512  # Maximum number of tokens for transformer models
BATCH_SIZE = 32

# Neutral Detection Thresholds (for enhanced neutral classification)
# The model uses these thresholds to determine when to classify as NEUTRAL:
# - If top prediction confidence < 80% AND neutral score > 20%, choose NEUTRAL
# - This ensures texts with significant neutral probability are properly classified
CONFIDENCE_THRESHOLD = 0.80  # Top prediction must be > 80% confident to override neutral
NEUTRAL_THRESHOLD = 0.20     # Neutral score must be > 20% to be considered

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
