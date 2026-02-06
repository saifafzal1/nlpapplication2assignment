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

#### On macOS:
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
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

### Method 1: Clone from GitHub (Recommended)

```bash
# Step 1: Clone the repository
git clone https://github.com/yourusername/nlpapplication2assignment.git

# Step 2: Navigate to project directory
cd nlpapplication2assignment

# Step 3: Create virtual environment
python -m venv venv

# Step 4: Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# Step 5: Upgrade pip (recommended)
python -m pip install --upgrade pip

# Step 6: Install dependencies
pip install -r requirements.txt

# Step 7: Download NLTK data (automatic on first run, but can be pre-downloaded)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
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

### Standard Execution

```bash
# Step 1: Ensure virtual environment is activated
# You should see (venv) prefix in your terminal

# Step 2: Run the Streamlit application
streamlit run app.py

# The application will automatically open in your default browser
# If not, manually navigate to: http://localhost:8501
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

### Issue 1: Python Command Not Found

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

### Issue 2: Virtual Environment Activation Fails

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

### Issue 3: Pip Install Errors

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

### Issue 4: Model Download Fails

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

### Issue 5: NLTK Data Download Issues

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

### Issue 6: Streamlit Not Found

```bash
# Verify installation
pip list | grep streamlit

# Reinstall if necessary
pip uninstall streamlit
pip install streamlit==1.28.1
```

### Issue 7: Port Already in Use

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

# Step 4: Create virtual environment
python3 -m venv venv

# Step 5: Activate virtual environment
source venv/bin/activate

# Step 6: Install dependencies
pip install -r requirements.txt

# Step 7: Run application
streamlit run app.py --server.headless true

# Step 8: Access via provided URL or port forwarding
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

### Step 1: Verify Installation
```bash
# Check Python version
python --version
# Should show: Python 3.10.x or higher

# Check installed packages
pip list
# Should show: streamlit, transformers, torch, nltk, etc.
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

**Last Updated**: February 2026
**Assignment**: BITS Pilani M.Tech AIML - NLP Applications Assignment 2 (PS-9)
