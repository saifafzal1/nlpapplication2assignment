# Submission Guidelines

This document provides complete submission guidelines for Assignment 2 - PS-9 (NLP Applications).

## Submission Checklist

### PART-A Deliverables

#### 1. Source Code Files
- [ ] `app.py` - Main Streamlit application (with detailed comments)
- [ ] `config.py` - Configuration settings (with detailed comments)
- [ ] `src/sentiment_analyzer.py` - Sentiment analysis module (with detailed comments)
- [ ] `src/preprocessor.py` - Text preprocessing module (with detailed comments)
- [ ] `src/visualizer.py` - Visualization module (with detailed comments)
- [ ] `src/__init__.py` - Package initialization
- [ ] `requirements.txt` - Python dependencies
- [ ] `Dockerfile` - Container configuration
- [ ] `.gitignore` - Git exclusions

#### 2. Documentation Files
- [ ] `README.md` - Project overview and setup instructions
- [ ] `execution.md` - Step-by-step execution guide
- [ ] `docs/report.md` - Implementation report with design choices and challenges

#### 3. Screenshots
- [ ] `screenshots/01_home_interface.png` - Home screen
- [ ] `screenshots/02_text_input.png` - Text input feature
- [ ] `screenshots/03_file_upload.png` - File upload feature
- [ ] `screenshots/04_preprocessing.png` - Preprocessing visualization
- [ ] `screenshots/05_results.png` - Sentiment analysis results
- [ ] `screenshots/06_osha_lab_portal.png` - OSHA Lab credentials (MANDATORY)

#### 4. Task B - Enhancement Plan
- [ ] `docs/task_b_enhancement_plan.md` - Markdown version
- [ ] `docs/task_b_enhancement_plan.pdf` - PDF version (MANDATORY)

### PART-B Deliverables

#### 5. Literature Survey
- [ ] `docs/literature_survey.md` - Markdown version
- [ ] `docs/literature_survey.pdf` - PDF version (MANDATORY)

### GitHub Repository
- [ ] Repository created: `nlpapplication2assignment`
- [ ] All code files committed
- [ ] README.md visible on GitHub
- [ ] Repository URL documented

---

## Submission Package Structure

Your final submission should be organized as follows:

```
nlpapplication2assignment/
├── Source Code
│   ├── app.py
│   ├── config.py
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── src/
│   │   ├── __init__.py
│   │   ├── sentiment_analyzer.py
│   │   ├── preprocessor.py
│   │   └── visualizer.py
│   └── sample_texts/
│
├── Documentation
│   ├── README.md
│   ├── execution.md
│   └── docs/
│       ├── report.md
│       ├── task_b_enhancement_plan.md
│       ├── task_b_enhancement_plan.pdf
│       ├── literature_survey.md
│       └── literature_survey.pdf
│
└── Screenshots
    ├── 01_home_interface.png
    ├── 02_text_input.png
    ├── 03_file_upload.png
    ├── 04_preprocessing.png
    ├── 05_results.png
    └── 06_osha_lab_portal.png
```

---

## Pre-Submission Verification

### Step 1: Code Quality Check

```bash
# Verify all files have proper comments
# Check each .py file for:
# - Module docstrings
# - Function docstrings
# - Inline comments for complex logic
```

### Step 2: Functionality Test

```bash
# Run the application
streamlit run app.py

# Test all features:
# 1. Text input analysis
# 2. File upload analysis
# 3. Preprocessing visualization
# 4. All visualizations render correctly
# 5. No errors in console
```

### Step 3: Documentation Review

```bash
# Verify README.md:
# - Clear project description
# - Complete installation instructions
# - Usage examples

# Verify execution.md:
# - Step-by-step instructions
# - Troubleshooting section
# - Works on fresh system

# Verify report.md:
# - Design choices explained
# - Challenges documented
# - Screenshots included with captions
```

### Step 4: Screenshot Quality Check

Ensure all screenshots:
- Are clear and readable (1920x1080 or higher)
- Show relevant application features
- Include UI elements and results
- Have descriptive filenames
- OSHA Lab screenshot shows credentials and timestamp

### Step 5: PDF Conversion

```bash
# Convert Markdown to PDF using one of these methods:

# Method 1: Using Pandoc (Recommended)
pandoc docs/task_b_enhancement_plan.md -o docs/task_b_enhancement_plan.pdf
pandoc docs/literature_survey.md -o docs/literature_survey.pdf

# Method 2: Using online converter
# Upload .md files to: https://www.markdowntopdf.com/

# Method 3: Using VS Code extension
# Install "Markdown PDF" extension
# Right-click .md file → "Markdown PDF: Export (pdf)"
```

Verify PDFs:
- All formatting preserved
- Images/diagrams visible
- References formatted correctly
- Page numbers added
- Professional appearance

---

## GitHub Repository Setup

### Step 1: Create Repository

```bash
# Go to GitHub (https://github.com)
# Click "New Repository"
# Name: nlpapplication2assignment
# Description: Sentiment Analysis Application for BITS Pilani NLP Assignment
# Visibility: Public (or Private if allowed by instructor)
# Initialize with: None (we already have files)
```

### Step 2: Push Code to GitHub

```bash
# Navigate to project directory
cd nlpapplication2assignment

# Initialize git repository (if not already done)
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Sentiment Analysis Application

- Implemented DistilBERT-based sentiment analyzer
- Created Streamlit web interface
- Added NLTK preprocessing pipeline
- Included comprehensive documentation
- Added sample test files and screenshots"

# Add remote repository
git remote add origin https://github.com/yourusername/nlpapplication2assignment.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 3: Verify Repository

Visit your GitHub repository and verify:
- [ ] All files are visible
- [ ] README.md displays correctly on main page
- [ ] Code has proper syntax highlighting
- [ ] Folder structure is organized
- [ ] No sensitive information (passwords, API keys) committed

---

## OSHA Lab Requirements

### Taking the Required Screenshot

1. **Log in to BITS OSHA Cloud Lab**
   - Navigate to: https://osha.bits-pilani.ac.in (or provided URL)
   - Use your student credentials

2. **Run the Application**
   ```bash
   cd nlpapplication2assignment
   source venv/bin/activate
   streamlit run app.py
   ```

3. **Capture Screenshot**
   - Ensure visible elements:
     - OSHA Lab portal header/menu bar
     - Your student credentials/name
     - Application running successfully
     - Current date/timestamp
     - Sentiment analysis results

4. **Save Screenshot**
   - Format: PNG or JPEG
   - Resolution: Minimum 1920x1080
   - Filename: `06_osha_lab_portal.png`
   - Location: `screenshots/` directory

---

## Markdown to PDF Conversion Guide

### Using Pandoc (Recommended)

#### Installation:

**Windows:**
```bash
# Download installer from: https://pandoc.org/installing.html
# Or use Chocolatey:
choco install pandoc
```

**macOS:**
```bash
brew install pandoc
```

**Linux:**
```bash
sudo apt-get install pandoc
```

#### Conversion Commands:

```bash
# Basic conversion
pandoc input.md -o output.pdf

# With table of contents
pandoc input.md -o output.pdf --toc

# With custom styling
pandoc input.md -o output.pdf --pdf-engine=xelatex

# For Task B Enhancement Plan
pandoc docs/task_b_enhancement_plan.md -o docs/task_b_enhancement_plan.pdf \
  --toc \
  --pdf-engine=xelatex \
  -V geometry:margin=1in

# For Literature Survey
pandoc docs/literature_survey.md -o docs/literature_survey.pdf \
  --toc \
  --pdf-engine=xelatex \
  -V geometry:margin=1in \
  --number-sections
```

### Using Online Tools

1. **Markdown to PDF**
   - URL: https://www.markdowntopdf.com/
   - Upload .md file
   - Click "Convert"
   - Download PDF

2. **Dillinger**
   - URL: https://dillinger.io/
   - Paste markdown content
   - Export as PDF

### Using VS Code

1. Install "Markdown PDF" extension
2. Open .md file
3. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
4. Type "Markdown PDF: Export (pdf)"
5. Select output location

---

## Final Submission Format

### Option 1: GitHub Repository Link (Recommended)

Submit the GitHub repository URL:
```
https://github.com/yourusername/nlpapplication2assignment
```

Ensure the repository contains:
- All source code
- All documentation (including PDFs)
- All screenshots (including OSHA Lab screenshot)

### Option 2: ZIP File

If submitting as ZIP:

```bash
# Create submission ZIP
# Exclude virtual environment and cache
zip -r NLP_Assignment2_PS9_<YourName>.zip \
  nlpapplication2assignment \
  -x "*/venv/*" "*/node_modules/*" "*/__pycache__/*" "*/.git/*"
```

ZIP should contain:
- Source code with comments
- Documentation (Markdown + PDFs)
- Screenshots (including OSHA Lab)
- README.md with setup instructions

---

## Grading Criteria Reference

### PART-A (10 marks)

**Task A - Web Interface (4 marks)**
- User interface for text input: ✓
- File upload functionality: ✓
- Sentiment visualization (charts): ✓
- Color-coded labels: ✓

**Task A - Sentiment Analysis (4 marks)**
- NLP model integration (DistilBERT): ✓
- Text preprocessing (NLTK): ✓
- Sentiment prediction: ✓
- Confidence scores: ✓

**Task B - Enhancement Plan (2 marks)**
- Detailed documentation: Check PDF
- Step-by-step process: Check PDF
- Real-time feedback design: Check PDF
- Model refinement strategy: Check PDF

### PART-B (5 marks)

**Literature Survey**
- Comprehensive review: Check PDF
- Multimodal sentiment analysis focus: ✓
- Recent research (2020-2026): Check PDF
- Proper citations: Check PDF

---

## Important Reminders

### Before Submission:

1. **Test on Fresh System**
   - Follow execution.md on a clean environment
   - Verify all dependencies install correctly
   - Confirm application runs without errors

2. **Code Comments**
   - Every function has docstring
   - Complex logic has inline comments
   - Module-level documentation present

3. **Screenshots**
   - All 6 screenshots captured
   - Clear and high-resolution
   - OSHA Lab credentials visible

4. **PDFs Generated**
   - Task B enhancement plan converted to PDF
   - Literature survey converted to PDF
   - Both PDFs are well-formatted

5. **GitHub Repository**
   - Code pushed successfully
   - README displays correctly
   - No broken links

---

## Submission Deadline

**No extensions on deadline** (as per assignment instructions)

Ensure you submit:
- Well before the deadline
- After thorough testing
- With all required components

---

## Contact Information

For any queries regarding submission:
- **Course LF**: Vasugi I
- **Email**: vasugii@wilp.bits-pilani.ac.in

---

## Post-Submission Checklist

After submitting, verify:
- [ ] Submission confirmation received
- [ ] GitHub repository is accessible
- [ ] All files are included
- [ ] PDFs are readable
- [ ] Screenshots are visible
- [ ] No errors in code files

---

**Good luck with your submission!**

**Assignment**: BITS Pilani M.Tech AIML - NLP Applications Assignment 2 (PS-9)
**Semester**: S1-25_AIMLCZG519
