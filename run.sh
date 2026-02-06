#!/bin/bash
# Startup script for Sentiment Analysis Application

# Configure pyenv
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Activate virtual environment
source venv/bin/activate

# Run Streamlit application
streamlit run app.py
