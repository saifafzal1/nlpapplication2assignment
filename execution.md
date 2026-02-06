# Execution Instructions

This document provides detailed step-by-step instructions for running the Sentiment Analysis Application on a fresh system. Follow these instructions carefully to ensure successful deployment.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Fresh System Setup](#fresh-system-setup)
3. [Installation Steps](#installation-steps)
4. [Running the Application](#running-the-application)
5. [Docker Deployment](#docker-deployment)
6. [Troubleshooting](#troubleshooting)
7. [OSHA Lab Specific Instructions](#osha-lab-specific-instructions)

---

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python**: Version 3.10 or higher
- **RAM**: 2GB minimum, 4GB recommended
- **Disk Space**: 3GB free space
- **Internet**: Required for initial model download

### Recommended Software
- **Code Editor**: VS Code, PyCharm, or any text editor
- **Terminal**: Command Prompt (Windows), Terminal (macOS/Linux)
- **Git**: For cloning the repository

---

## Fresh System Setup

### Step 1: Install Python

#### On Windows:
1. Download Python 3.10+ from [python.org](https://www.python.org/downloads/)
2. Run the installer
3. **IMPORTANT**: Check "Add Python to PATH" during installation
4. Verify installation:
   ```bash
   python --version
   pip --version
   ```

#### On macOS (Recommended: Using pyenv):
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install pyenv for Python version management
brew install pyenv

# Configure pyenv (add to ~/.zshrc or ~/.bash_profile)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc

# Restart terminal or source config
source ~/.zshrc

# Verify pyenv installation
pyenv --version
```

#### Alternative macOS (Direct Python install):
```bash
# Install Python directly via Homebrew
brew install python@3.10

# Verify installation
python3 --version
pip3 --version
```

#### On Linux (Ubuntu/Debian):
```bash
# Update package list
sudo apt update

# Install Python 3.10
sudo apt install python3.10 python3-pip python3-venv

# Verify installation
python3 --version
pip3 --version
```

### Step 2: Install Git (Optional but Recommended)

#### On Windows:
Download and install from [git-scm.com](https://git-scm.com/download/win)

#### On macOS:
```bash
brew install git
```

#### On Linux:
```bash
sudo apt install git
```

---

## Installation Steps

### Method 1: Clone from GitHub with pyenv (Recommended for macOS)

```bash
# Step 1: Clone the repository
git clone https://github.com/yourusername/nlpapplication2assignment.git

# Step 2: Navigate to project directory
cd nlpapplication2assignment

# Step 3: Install Python 3.10 using pyenv
# List available versions
pyenv install --list | grep "^\s*3\.10\."

# Install Python 3.10.19 (latest 3.10 version)
pyenv install 3.10.19

# Set Python 3.10.19 for this project
pyenv local 3.10.19

# Verify Python version
python --version
# Should show: Python 3.10.19

# Step 4: Create virtual environment
python -m venv venv

# Step 5: Activate virtual environment
source venv/bin/activate

# Step 6: Upgrade pip (recommended)
python -m pip install --upgrade pip

# Step 7: Install dependencies
pip install -r requirements.txt

# Step 8: Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('averaged_perceptron_tagger')"

# Step 9: Verify installation
python -c "import streamlit; import transformers; import nltk; print('All imports successful!')"
```

### Method 1b: Clone from GitHub (Windows/Linux)

```bash
# Step 1: Clone the repository
git clone https://github.com/yourusername/nlpapplication2assignment.git

# Step 2: Navigate to project directory
cd nlpapplication2assignment

# Step 3: Create virtual environment with Python 3.10
python3.10 -m venv venv

# Step 4: Activate virtual environment
# On Windows:
venv\Scripts\activate

# On Linux:
source venv/bin/activate

# Step 5: Upgrade pip
python -m pip install --upgrade pip

# Step 6: Install dependencies
pip install -r requirements.txt

# Step 7: Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('averaged_perceptron_tagger')"
```

### Method 2: Manual Setup (If Git is not available)

```bash
# Step 1: Download the project ZIP from GitHub
# Extract to a folder (e.g., nlpapplication2assignment)

# Step 2: Open terminal/command prompt in that folder
cd path/to/nlpapplication2assignment

# Step 3-7: Follow same steps as Method 1
```

---

## Running the Application

### Option 1: Quick Start Script (macOS/Linux - Easiest)

A convenient startup script has been provided:

```bash
# Navigate to project directory
cd /Users/nadiaashfaq/Documents/NLPAssignment2

# Run the application (script handles pyenv and venv activation)
./run.sh

# The application will automatically open in your default browser
# If not, manually navigate to: http://localhost:8501
```

### Option 2: Manual Execution (All Platforms)

```bash
# Step 1: Navigate to project directory
cd /path/to/nlpapplication2assignment

# Step 2: Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Step 3: Run the Streamlit application
streamlit run app.py

# The application will automatically open in your default browser
# If not, manually navigate to: http://localhost:8501
```

### Option 3: With pyenv Configuration (macOS)

```bash
# Step 1: Configure pyenv environment
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Step 2: Activate virtual environment
source venv/bin/activate

# Step 3: Run the Streamlit application
streamlit run app.py
```

### Custom Port (if 8501 is busy)

```bash
streamlit run app.py --server.port 8502
```

### Headless Mode (no browser auto-open)

```bash
streamlit run app.py --server.headless true
```

### Alternative Python Command (if `streamlit` command not found)

```bash
python -m streamlit run app.py
```

### First Run Experience

**Initial Launch (10-15 seconds):**
- DistilBERT model downloads automatically (~260MB)
- Model is cached in `~/.cache/huggingface/`
- Application starts after download completes
- Browser opens automatically at `http://localhost:8501`

**Subsequent Launches (2-3 seconds):**
- Model loads from cache
- Application starts immediately
- Much faster startup

**What You'll See:**
1. Terminal shows: "You can now view your Streamlit app in your browser"
2. Browser opens to the Sentiment Analysis Application
3. Ready to analyze text!

---

## Docker Deployment

### Prerequisites
- Install Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop/)

### Build and Run

```bash
# Step 1: Navigate to project directory
cd nlpapplication2assignment

# Step 2: Build Docker image (takes 5-10 minutes first time)
docker build -t sentiment-analyzer .

# Step 3: Run Docker container
docker run -p 8501:8501 sentiment-analyzer

# Step 4: Access application at http://localhost:8501
```

### Docker Compose (Alternative)

```bash
# If docker-compose.yml is provided
docker-compose up
```

---

## Troubleshooting

### Issue 1: Wrong Python Version (3.14+ instead of 3.10)

**Using pyenv (macOS/Linux):**
```bash
# Check current Python version
python --version

# If not 3.10.x, install and set Python 3.10
pyenv install 3.10.19
pyenv local 3.10.19

# Remove old virtual environment
rm -rf venv

# Create new virtual environment with correct Python version
python -m venv venv
source venv/bin/activate

# Verify version
python --version
# Should show: Python 3.10.19

# Reinstall dependencies
pip install -r requirements.txt
```

**Alternative (Download specific Python version):**
```bash
# Download Python 3.10.19 from python.org
# Install it
# Use specific version to create venv
python3.10 -m venv venv
```

### Issue 2: Python Command Not Found

**Windows:**
```bash
# Try using 'py' instead of 'python'
py -m venv venv
py -m pip install -r requirements.txt
```

**macOS/Linux:**
```bash
# Try using 'python3' instead of 'python'
python3 -m venv venv
python3 -m pip install -r requirements.txt
```

### Issue 3: pyenv Not Configured Properly

**Symptoms:** Python version doesn't change or pyenv command not found

**Solution:**
```bash
# Ensure pyenv is installed
brew install pyenv

# Add to shell configuration
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc

# Reload shell configuration
source ~/.zshrc

# Or restart terminal
exec "$SHELL"

# Verify pyenv is working
pyenv --version
pyenv versions
```

### Issue 4: Virtual Environment Activation Fails

**Windows PowerShell:**
If you get an execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\Activate.ps1
```

**Alternative (Windows):**
```bash
venv\Scripts\activate.bat
```

### Issue 5: Pip Install Errors

```bash
# Update pip first
python -m pip install --upgrade pip

# Install with verbose output to see errors
pip install -r requirements.txt -v

# If specific package fails, install individually
pip install streamlit
pip install transformers
pip install torch
```

### Issue 6: Model Download Fails

```bash
# Check internet connection
# Try downloading with Python directly:
python -c "from transformers import pipeline; pipeline('sentiment-analysis')"

# If behind proxy, set environment variables:
# Windows:
set HTTP_PROXY=http://proxy.server:port
set HTTPS_PROXY=http://proxy.server:port

# macOS/Linux:
export HTTP_PROXY=http://proxy.server:port
export HTTPS_PROXY=http://proxy.server:port
```

### Issue 7: NLTK Data Download Issues

```bash
# Manual download
python
>>> import nltk
>>> nltk.download('punkt')
>>> nltk.download('stopwords')
>>> nltk.download('wordnet')
>>> nltk.download('omw-1.4')
>>> exit()
```

### Issue 8: Streamlit Not Found

```bash
# Verify installation
pip list | grep streamlit

# Reinstall if necessary
pip uninstall streamlit
pip install streamlit==1.28.1
```

### Issue 9: Port Already in Use

```bash
# Check what's using port 8501
# Windows:
netstat -ano | findstr :8501

# macOS/Linux:
lsof -i :8501

# Use different port:
streamlit run app.py --server.port 8502
```

---

## OSHA Lab Specific Instructions

### Accessing OSHA Lab
1. Log in to BITS OSHA Cloud Lab portal
2. Navigate to your assigned virtual machine
3. Open terminal/command prompt

### Installation on OSHA Lab

```bash
# Step 1: Navigate to home directory
cd ~

# Step 2: Clone or upload project
git clone https://github.com/yourusername/nlpapplication2assignment.git
# OR upload ZIP and extract

# Step 3: Navigate to project
cd nlpapplication2assignment

# Step 4: Check Python version
python3 --version

# If Python 3.10+ is available:
python3 -m venv venv
source venv/bin/activate

# If Python 3.10 is not available, install pyenv:
# Install pyenv (if not already available)
curl https://pyenv.run | bash

# Configure pyenv
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Install Python 3.10.19
pyenv install 3.10.19
pyenv local 3.10.19

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Step 5: Upgrade pip
pip install --upgrade pip

# Step 6: Install dependencies
pip install -r requirements.txt

# Step 7: Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('averaged_perceptron_tagger')"

# Step 8: Run application (headless mode for server)
streamlit run app.py --server.headless true --server.port 8501

# Step 9: Access via provided URL
# The URL will be displayed in the terminal
# Example: http://osha-lab-server:8501
```

### Taking Screenshot for Submission

1. Ensure you're logged into OSHA Lab portal
2. Open the Sentiment Analysis application
3. Capture screenshot showing:
   - OSHA Lab portal header with your credentials
   - Application running successfully
   - Timestamp visible
4. Save as `screenshots/06_osha_lab_portal.png`

---

## Verification Steps

### Step 1: Verify Python Version
```bash
# Activate virtual environment first
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate  # Windows

# Check Python version
python --version
# Should show: Python 3.10.19 (or 3.10.x)

# If using pyenv, verify it's using the correct version
which python
# Should show: /Users/nadiaashfaq/Documents/NLPAssignment2/venv/bin/python
```

### Step 2: Verify Installed Packages
```bash
# Check key packages are installed
pip list | grep -E "streamlit|transformers|torch|nltk|plotly|pandas"

# Expected output:
# nltk                      3.8.1
# pandas                    2.1.3
# plotly                    5.18.0
# streamlit                 1.28.1
# torch                     2.1.2
# transformers              4.36.2

# Check all installed packages
pip list
```

### Step 3: Verify Imports
```bash
# Test that all critical modules can be imported
python -c "import streamlit; import transformers; import nltk; import plotly; import pandas; print('✓ All imports successful!')"

# Check versions
python -c "import streamlit; import transformers; print('Streamlit:', streamlit.__version__); print('Transformers:', transformers.__version__)"
```

### Step 2: Test Application Launch
```bash
streamlit run app.py
# Should start without errors
# Should show: "You can now view your Streamlit app in your browser"
```

### Step 3: Test Functionality
1. Open application in browser
2. Enter text: "I love this product!"
3. Click "Analyze Sentiment"
4. Verify:
   - Model loads successfully
   - Sentiment is detected as POSITIVE
   - Confidence score is displayed
   - Visualizations appear

### Step 4: Test File Upload
1. Click "File Upload" option
2. Upload `sample_texts/positive.txt`
3. Click "Analyze Sentiment"
4. Verify results appear correctly

---

## Performance Notes

### First Run
- **Time**: 10-15 seconds
- **Reason**: Model downloads (~260MB)
- **Location**: Cached in `~/.cache/huggingface/`

### Subsequent Runs
- **Time**: 2-3 seconds
- **Reason**: Model loaded from cache

### Analysis Speed
- **Per Text**: 50-200ms
- **Depends On**: Text length, CPU speed

---

## Environment Variables (Optional)

Create a `.env` file for custom configuration:

```bash
# .env
MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
MODEL_CACHE_DIR=./models
MAX_TEXT_LENGTH=512
CONFIDENCE_THRESHOLD=0.60
```

---

## Stopping the Application

### Terminal Method
Press `Ctrl + C` in the terminal where Streamlit is running

### Docker Method
```bash
# Find container ID
docker ps

# Stop container
docker stop <container_id>
```

---

## Uninstallation

```bash
# Step 1: Deactivate virtual environment
deactivate

# Step 2: Delete project folder
# Windows:
rmdir /s nlpapplication2assignment

# macOS/Linux:
rm -rf nlpapplication2assignment

# Step 3: Clear model cache (optional)
# Windows:
rmdir /s %USERPROFILE%\.cache\huggingface

# macOS/Linux:
rm -rf ~/.cache/huggingface
```

---

## Additional Resources

- **Streamlit Documentation**: https://docs.streamlit.io
- **Hugging Face Transformers**: https://huggingface.co/docs/transformers
- **NLTK Documentation**: https://www.nltk.org
- **Python Virtual Environments**: https://docs.python.org/3/tutorial/venv.html

---

## Support

If you encounter issues not covered in this guide:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review error messages carefully
3. Search for the error online (Stack Overflow, GitHub Issues)
4. Contact course instructor: vasugii@wilp.bits-pilani.ac.in

---

## Quick Reference Guide

### Daily Use Commands

**Start the Application:**
```bash
cd /Users/nadiaashfaq/Documents/NLPAssignment2
./run.sh
```

**OR Manual Start:**
```bash
cd /Users/nadiaashfaq/Documents/NLPAssignment2
source venv/bin/activate
streamlit run app.py
```

**Stop the Application:**
```
Press Ctrl + C in the terminal
```

### Common Commands

**Check Python Version:**
```bash
python --version
```

**Check Installed Packages:**
```bash
pip list
```

**Update a Package:**
```bash
pip install --upgrade package-name
```

**Deactivate Virtual Environment:**
```bash
deactivate
```

**Reinstall All Dependencies:**
```bash
pip install -r requirements.txt --force-reinstall
```

### File Locations

| Item | Location |
|------|----------|
| Project Directory | `/Users/nadiaashfaq/Documents/NLPAssignment2` |
| Virtual Environment | `/Users/nadiaashfaq/Documents/NLPAssignment2/venv` |
| Python Version File | `/Users/nadiaashfaq/Documents/NLPAssignment2/.python-version` |
| Model Cache | `~/.cache/huggingface/` |
| NLTK Data | `~/nltk_data/` |
| Startup Script | `/Users/nadiaashfaq/Documents/NLPAssignment2/run.sh` |

### Quick Checks

**Is Virtual Environment Active?**
```bash
# Your prompt should show (venv)
# OR check:
which python
# Should show path to venv/bin/python
```

**Is Application Running?**
```bash
# Check if port 8501 is in use:
lsof -i :8501
```

**View Application Logs:**
```bash
# Logs are shown in the terminal where you ran streamlit
# For debugging, run with verbose flag:
streamlit run app.py --logger.level=debug
```

---

## Summary

You now have a complete working environment with:
- ✓ Python 3.10.19 (managed by pyenv)
- ✓ Virtual environment with all dependencies
- ✓ NLTK data downloaded
- ✓ DistilBERT model (downloads on first run)
- ✓ Quick start script (run.sh)

**To start working:**
```bash
cd /Users/nadiaashfaq/Documents/NLPAssignment2
./run.sh
```

**For help:**
- Check [README.md](README.md) for project overview
- Check [submission.md](submission.md) for submission guidelines
- Check [docs/report.md](docs/report.md) for implementation details

---

**Last Updated**: February 2026
**Assignment**: BITS Pilani M.Tech AIML - NLP Applications Assignment 2 (PS-9)
**Python Version**: 3.10.19 (via pyenv)
**Environment**: Fully configured and tested
