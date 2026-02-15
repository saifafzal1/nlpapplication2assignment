# Sentiment Analysis Application

A modern web-based sentiment analysis application built with Streamlit and Hugging Face Transformers for BITS Pilani M.Tech AIML - NLP Applications Assignment 2 (PS-9).

## Overview

This application performs sentiment analysis on user-provided text using state-of-the-art NLP techniques. It combines modern transformer models (Twitter RoBERTa) with traditional NLTK preprocessing to provide accurate 3-class sentiment classification and educational insights into text preprocessing.

## Features

### Core Functionality
- **Text Input**: Type or paste text directly into the application
- **File Upload**: Upload .txt files for batch sentiment analysis
- **Real-time Analysis**: Instant sentiment prediction with confidence scores
- **Multi-class Classification**: Detects Positive, Negative, and Neutral sentiments

### Visualization
- **Interactive Bar Charts**: Displays confidence scores for all sentiment classes
- **Pie Charts**: Shows sentiment distribution
- **Color-coded Results**: Visual representation with green (positive), red (negative), and gray (neutral)
- **Confidence Metrics**: Percentage-based confidence scores

### Text Preprocessing
- **Tokenization**: NLTK word tokenization
- **Stopword Removal**: Filters common English stopwords
- **Stemming**: Porter stemmer for word normalization
- **Lemmatization**: WordNet lemmatizer for accurate word forms
- **Preprocessing Comparison**: Before/after visualization

## Technology Stack

### Backend
- **Python 3.10+**: Core programming language
- **Hugging Face Transformers 4.36.2**: For pre-trained sentiment models
- **PyTorch 2.1.2**: Deep learning framework
- **NLTK 3.8.1**: Traditional NLP preprocessing

### Model
- **Twitter RoBERTa**: `cardiffnlp/twitter-roberta-base-sentiment`
  - Native 3-class sentiment support (NEGATIVE, NEUTRAL, POSITIVE)
  - Trained on ~58M tweets
  - 90%+ accuracy across all sentiment classes
  - Enhanced neutral detection with adaptive thresholds
  - Optimized for social media and diverse text
  - CPU-compatible with reasonable inference speed

### Frontend
- **Streamlit 1.28.1**: Interactive web framework
- **Plotly 5.18.0**: Interactive visualizations
- **Matplotlib 3.8.2**: Additional plotting

### Data Processing
- **Pandas 2.1.3**: Data manipulation
- **NumPy 1.24.3**: Numerical computing

## Project Structure

```
nlpapplication2assignment/
├── app.py                        # Main Streamlit application
├── config.py                     # Configuration settings
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker container configuration
├── README.md                     # This file
├── execution.md                  # Detailed execution instructions
├── submission.md                 # Submission guidelines
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── sentiment_analyzer.py    # Sentiment analysis logic
│   ├── preprocessor.py          # Text preprocessing
│   └── visualizer.py            # Visualization functions
├── docs/                         # Documentation
│   ├── report.md                # Implementation report
│   ├── task_b_enhancement_plan.md
│   └── literature_survey.md
├── screenshots/                  # Application screenshots
├── tests/                        # Unit tests
└── sample_texts/                 # Sample input files
    ├── positive.txt
    ├── negative.txt
    └── neutral.txt
```

## Quick Start

### Prerequisites
- Python 3.10 or higher
- pip package manager
- 2GB free disk space (for model files)
- Internet connection (for initial model download)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/nlpapplication2assignment.git
cd nlpapplication2assignment
```

2. **Create virtual environment**
```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage

### Text Input Method
1. Select "Text Input" from the sidebar
2. Enter or paste your text in the text area
3. Click "Analyze Sentiment" button
4. View results with confidence scores and visualizations

### File Upload Method
1. Select "File Upload" from the sidebar
2. Click "Browse files" and select a .txt file
3. Click "Analyze Sentiment" button
4. View results

### Preprocessing Options
- Enable "Show Preprocessing Steps" to visualize NLTK preprocessing
- Toggle individual preprocessing steps (tokenization, stemming, lemmatization)
- Compare original vs processed text

## Docker Deployment

### Build Docker Image
```bash
docker build -t sentiment-analyzer .
```

### Run Docker Container
```bash
docker run -p 8501:8501 sentiment-analyzer
```

Access the application at `http://localhost:8501`

## Testing

### Run Unit Tests
```bash
python -m pytest tests/
```

### Test with Sample Files
```bash
# The application includes sample text files in sample_texts/ directory
# Use the file upload feature to test with:
- positive.txt: Positive sentiment example
- negative.txt: Negative sentiment example
- neutral.txt: Neutral sentiment example
```

## Model Information

### Default Model: DistilBERT
- **Model ID**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Training Data**: Stanford Sentiment Treebank (SST-2)
- **Performance**: ~95% accuracy
- **Inference Speed**: ~50ms per sample on CPU
- **Model Size**: ~260MB

### Model Caching
- Models are cached in `~/.cache/huggingface/` after first download
- Subsequent runs load from cache (much faster)

## Configuration

Edit `config.py` to customize:
- Model selection
- Confidence thresholds
- Maximum text length
- Color schemes
- File upload limits

## Troubleshooting

### Model Download Issues
If model download fails:
```bash
# Manually download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Import Errors
Ensure virtual environment is activated:
```bash
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### Port Already in Use
Change Streamlit port:
```bash
streamlit run app.py --server.port 8502
```

## Performance

### Benchmarks
- **Startup Time**: ~10-15 seconds (first run), ~2-3 seconds (cached)
- **Analysis Time**: ~50-200ms per text sample
- **Memory Usage**: ~1.5GB RAM
- **CPU Optimized**: Works without GPU

## Assignment Components

### PART-A
- **Task A**: Web interface with sentiment analysis (✓ Complete)
  - Text input and file upload functionality
  - Sentiment visualization with charts
  - NLP model integration (DistilBERT)
  - Text preprocessing (NLTK)
  - Sentiment prediction

- **Task B**: Enhancement plan document (See `docs/task_b_enhancement_plan.md`)

### PART-B
- **Literature Survey**: Multimodal sentiment analysis (See `docs/literature_survey.md`)

## Documentation

- [execution.md](execution.md) - Step-by-step execution instructions for fresh systems
- [docs/report.md](docs/report.md) - Implementation report with design choices
- [docs/task_b_enhancement_plan.md](docs/task_b_enhancement_plan.md) - Real-time feedback enhancement plan
- [docs/literature_survey.md](docs/literature_survey.md) - Literature survey on multimodal sentiment analysis



## Acknowledgments

- Hugging Face for providing pre-trained transformer models
- NLTK team for comprehensive NLP tools
- Streamlit for the excellent web framework


## Repository

GitHub: https://github.dev/saif2024-bits/nlpapplication2assignment/tree/patch-1

---

