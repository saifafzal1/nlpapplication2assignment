# Task B: Enhancement Plan
## Real-Time Feedback Feature for Sentiment Analysis Application

**Course**: NLP Applications (S1-25_AIMLCZG519)
**Assignment**: 2 - PS-9 (Task B)
**Institution**: BITS Pilani M.Tech AIML

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Current System Overview](#current-system-overview)
3. [Enhancement Objectives](#enhancement-objectives)
4. [System Architecture](#system-architecture)
5. [Detailed Implementation Steps](#detailed-implementation-steps)
6. [Technical Specifications](#technical-specifications)
7. [Model Refinement Strategy](#model-refinement-strategy)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Deployment Plan](#deployment-plan)
10. [Timeline and Resources](#timeline-and-resources)

---

## Executive Summary

This document outlines a comprehensive plan to enhance the existing Sentiment Analysis Application with a **real-time feedback feature** that enables users to provide immediate input on the accuracy of sentiment analysis results. The collected feedback will be used to continuously refine and improve the underlying DistilBERT model through active learning and fine-tuning techniques.

### Key Enhancement Features:
- Interactive feedback buttons (thumbs up/down)
- Detailed feedback forms for incorrect predictions
- Feedback database with SQLite
- Automated model retraining pipeline
- A/B testing framework
- Performance monitoring dashboard

### Expected Benefits:
- Improved model accuracy through domain adaptation
- Better handling of edge cases and ambiguous sentiments
- User-specific customization potential
- Continuous learning from real-world usage
- Measurable improvement metrics

---

## Current System Overview

### Existing Functionality

```
Input Text → DistilBERT Model → Sentiment Prediction → Display Results
```

**Current Features:**
- Text input or file upload
- Pre-trained DistilBERT sentiment analysis
- Visualization of confidence scores
- Static model (no learning from usage)

**Limitations:**
- No user feedback mechanism
- Model doesn't improve from mistakes
- No domain adaptation
- No user-specific preferences
- Cannot handle specialized vocabulary

---

## Enhancement Objectives

### Primary Goals

1. **Collect User Feedback**
   - Enable users to rate prediction accuracy
   - Gather corrected labels for misclassifications
   - Store feedback with context

2. **Improve Model Performance**
   - Fine-tune model on feedback data
   - Reduce misclassification errors
   - Adapt to domain-specific language

3. **Enable Continuous Learning**
   - Automated retraining pipeline
   - Incremental model updates
   - Version control for models

4. **Measure Impact**
   - Track accuracy improvements
   - Monitor user satisfaction
   - Compare model versions

### Success Criteria

- **User Engagement**: 30%+ of users provide feedback
- **Accuracy Improvement**: 2-5% increase in accuracy after refinement
- **Response Time**: Feedback collection adds <100ms overhead
- **Data Quality**: 80%+ feedback samples are valid

---

## System Architecture

### Enhanced Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                   Web Interface (Streamlit)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │ Text Input   │  │ File Upload  │  │ Feedback Buttons    │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬──────────────┘  │
└─────────┼──────────────────┼─────────────────┼─────────────────┘
          │                  │                 │
          └──────────────────┼─────────────────┘
                             │
                   ┌─────────▼──────────┐
                   │  Sentiment Analyzer │
                   │   (DistilBERT)     │
                   └─────────┬──────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    ┌──────────────────┐         ┌──────────────────┐
    │ Display Results  │         │ Feedback Handler │
    │ + Feedback UI    │         │ (New Component)  │
    └──────────────────┘         └────────┬─────────┘
                                          │
                                          ▼
                                 ┌─────────────────┐
                                 │ SQLite Database │
                                 │ - feedback_data │
                                 │ - user_sessions │
                                 │ - model_metrics │
                                 └────────┬────────┘
                                          │
                                          ▼
                            ┌─────────────────────────┐
                            │ Retraining Pipeline     │
                            │ (Scheduled/Triggered)   │
                            │ - Data validation       │
                            │ - Model fine-tuning     │
                            │ - Evaluation            │
                            │ - Deployment            │
                            └─────────────────────────┘
```

### Component Interactions

```
User → Provides Feedback → Stored in Database →
Triggers Retraining (when threshold met) →
Fine-tuned Model → Deployed → Better Predictions
```

---

## Detailed Implementation Steps

### Step 1: Database Design and Setup

**Objective**: Create database schema to store feedback data

**Implementation**:

```python
# src/database.py
import sqlite3
from datetime import datetime
import hashlib

class FeedbackDatabase:
    def __init__(self, db_path="data/feedback.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Feedback table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            predicted_label TEXT NOT NULL,
            predicted_confidence REAL NOT NULL,
            user_feedback TEXT, -- 'correct', 'incorrect'
            corrected_label TEXT, -- User's suggested label
            user_comment TEXT,
            session_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            text_hash TEXT UNIQUE -- Prevent duplicates
        )
        ''')

        # Model versions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_versions (
            version_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            training_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            training_samples INTEGER,
            accuracy REAL,
            precision_positive REAL,
            precision_negative REAL,
            recall_positive REAL,
            recall_negative REAL,
            f1_score REAL,
            is_active BOOLEAN DEFAULT 0
        )
        ''')

        # User sessions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            feedback_count INTEGER DEFAULT 0
        )
        ''')

        conn.commit()
        conn.close()

    def add_feedback(self, text, predicted_label, predicted_confidence,
                    user_feedback, corrected_label=None, comment=None,
                    session_id=None):
        """Add user feedback to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Generate text hash to avoid duplicates
        text_hash = hashlib.md5(text.encode()).hexdigest()

        try:
            cursor.execute('''
            INSERT INTO feedback
            (text, predicted_label, predicted_confidence, user_feedback,
             corrected_label, user_comment, session_id, text_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (text, predicted_label, predicted_confidence, user_feedback,
                  corrected_label, comment, session_id, text_hash))

            conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Duplicate entry
            return False
        finally:
            conn.close()

    def get_feedback_for_training(self, limit=None):
        """Retrieve feedback data for model retraining"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = '''
        SELECT text, corrected_label
        FROM feedback
        WHERE user_feedback = 'incorrect' AND corrected_label IS NOT NULL
        '''
        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()

        return results
```

**Database Schema**:

| Table | Columns | Purpose |
|-------|---------|---------|
| feedback | id, text, predicted_label, predicted_confidence, user_feedback, corrected_label, user_comment, session_id, timestamp, text_hash | Store all user feedback |
| model_versions | version_id, model_name, training_date, training_samples, accuracy, precision, recall, f1_score, is_active | Track model performance |
| sessions | session_id, start_time, feedback_count | Track user sessions |

---

### Step 2: Frontend Feedback UI Components

**Objective**: Add interactive feedback buttons to the results display

**Implementation**:

```python
# In app.py, add after sentiment results

def display_feedback_section(text, predicted_label, confidence):
    """Display feedback collection UI"""
    st.markdown("---")
    st.subheader("Was this prediction helpful?")

    col1, col2, col3 = st.columns([1, 1, 3])

    with col1:
        if st.button("👍 Correct", key="feedback_correct"):
            save_feedback(
                text=text,
                predicted_label=predicted_label,
                confidence=confidence,
                user_feedback="correct"
            )
            st.success("Thank you for your feedback!")

    with col2:
        if st.button("👎 Incorrect", key="feedback_incorrect"):
            st.session_state['show_correction_form'] = True

    # Show correction form if user said prediction was incorrect
    if st.session_state.get('show_correction_form', False):
        with st.expander("Help us improve", expanded=True):
            st.markdown("**What should the correct sentiment be?**")

            corrected_label = st.radio(
                "Correct sentiment:",
                options=["POSITIVE", "NEGATIVE", "NEUTRAL"],
                index=0,
                key="corrected_label"
            )

            user_comment = st.text_area(
                "Additional comments (optional):",
                placeholder="Tell us why this prediction was incorrect...",
                key="user_comment"
            )

            if st.button("Submit Correction", key="submit_correction"):
                save_feedback(
                    text=text,
                    predicted_label=predicted_label,
                    confidence=confidence,
                    user_feedback="incorrect",
                    corrected_label=corrected_label,
                    comment=user_comment
                )
                st.success("Thank you! Your correction will help improve our model.")
                st.session_state['show_correction_form'] = False
                st.rerun()

def save_feedback(text, predicted_label, confidence, user_feedback,
                 corrected_label=None, comment=None):
    """Save feedback to database"""
    db = FeedbackDatabase()
    session_id = st.session_state.get('session_id', str(uuid.uuid4()))
    st.session_state['session_id'] = session_id

    db.add_feedback(
        text=text,
        predicted_label=predicted_label,
        predicted_confidence=confidence,
        user_feedback=user_feedback,
        corrected_label=corrected_label,
        comment=comment,
        session_id=session_id
    )
```

**UI Design**:

```
┌─────────────────────────────────────────────┐
│         Sentiment Analysis Result           │
│                                             │
│         Sentiment: POSITIVE ✓               │
│         Confidence: 92.5%                   │
│                                             │
│         [Bar Chart]  [Pie Chart]           │
│                                             │
├─────────────────────────────────────────────┤
│  Was this prediction helpful?               │
│                                             │
│  [👍 Correct]  [👎 Incorrect]              │
│                                             │
│  ▼ Help us improve (if incorrect clicked)  │
│  ┌───────────────────────────────────────┐ │
│  │ What should the correct sentiment be? │ │
│  │ ○ POSITIVE                            │ │
│  │ ● NEGATIVE                            │ │
│  │ ○ NEUTRAL                             │ │
│  │                                       │ │
│  │ Additional comments:                  │ │
│  │ ┌───────────────────────────────────┐ │ │
│  │ │ [User's text explanation]         │ │ │
│  │ └───────────────────────────────────┘ │ │
│  │                                       │ │
│  │ [Submit Correction]                   │ │
│  └───────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

---

### Step 3: Feedback Validation and Quality Control

**Objective**: Ensure collected feedback is valid and useful

**Implementation**:

```python
# src/feedback_validator.py

class FeedbackValidator:
    def __init__(self):
        self.min_text_length = 5
        self.max_text_length = 5000

    def validate_feedback(self, text, predicted_label, corrected_label):
        """Validate feedback data quality"""
        issues = []

        # Check text length
        if len(text) < self.min_text_length:
            issues.append("Text too short")

        if len(text) > self.max_text_length:
            issues.append("Text too long")

        # Check if correction is different from prediction
        if predicted_label == corrected_label:
            issues.append("Correction same as prediction")

        # Check for valid sentiment labels
        valid_labels = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
        if corrected_label not in valid_labels:
            issues.append("Invalid corrected label")

        return len(issues) == 0, issues

    def filter_quality_feedback(self, feedback_list):
        """Filter high-quality feedback for training"""
        quality_feedback = []

        for text, label in feedback_list:
            # Skip very short or very long texts
            if len(text.split()) < 3 or len(text.split()) > 500:
                continue

            # Add other quality checks here
            quality_feedback.append((text, label))

        return quality_feedback
```

---

### Step 4: Model Fine-Tuning Pipeline

**Objective**: Retrain model on collected feedback data

**Implementation**:

```python
# src/model_trainer.py

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch

class SentimentModelTrainer:
    def __init__(self, base_model="distilbert-base-uncased-finetuned-sst-2-english"):
        self.base_model = base_model
        self.model = None
        self.tokenizer = None

    def prepare_training_data(self, feedback_data):
        """Convert feedback data to training format"""
        texts = []
        labels = []

        label_map = {"POSITIVE": 1, "NEGATIVE": 0, "NEUTRAL": 2}

        for text, label in feedback_data:
            texts.append(text)
            labels.append(label_map.get(label, 2))

        # Create Hugging Face dataset
        dataset = Dataset.from_dict({
            "text": texts,
            "label": labels
        })

        return dataset

    def tokenize_data(self, dataset):
        """Tokenize dataset"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset

    def fine_tune_model(self, train_dataset, output_dir="models/fine_tuned"):
        """Fine-tune model on feedback data"""

        # Load base model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=3  # POSITIVE, NEGATIVE, NEUTRAL
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,  # Use validation split in production
        )

        # Train
        trainer.train()

        # Save model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        return output_dir

    def retrain_with_feedback(self, min_samples=100):
        """Complete retraining pipeline"""

        # Step 1: Get feedback data
        db = FeedbackDatabase()
        feedback_data = db.get_feedback_for_training()

        if len(feedback_data) < min_samples:
            print(f"Not enough feedback samples ({len(feedback_data)}/{min_samples})")
            return None

        # Step 2: Validate feedback
        validator = FeedbackValidator()
        quality_feedback = validator.filter_quality_feedback(feedback_data)

        # Step 3: Prepare dataset
        dataset = self.prepare_training_data(quality_feedback)
        tokenized_dataset = self.tokenize_data(dataset)

        # Step 4: Fine-tune model
        output_dir = self.fine_tune_model(tokenized_dataset)

        print(f"Model fine-tuned successfully. Saved to: {output_dir}")
        return output_dir
```

---

### Step 5: A/B Testing Framework

**Objective**: Compare new fine-tuned model with original model

**Implementation**:

```python
# src/ab_testing.py

import random
from datetime import datetime

class ABTester:
    def __init__(self):
        self.model_a = SentimentAnalyzer(model_name="original_model")
        self.model_b = SentimentAnalyzer(model_name="fine_tuned_model")
        self.test_ratio = 0.5  # 50% users get each model

    def assign_model(self, session_id):
        """Assign user to model A or B"""
        # Use hash for consistent assignment
        hash_value = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
        return "A" if (hash_value % 100) < (self.test_ratio * 100) else "B"

    def get_model(self, session_id):
        """Get appropriate model for user"""
        assignment = self.assign_model(session_id)
        return self.model_a if assignment == "A" else self.model_b

    def log_result(self, session_id, model_version, text, prediction, feedback):
        """Log A/B test results"""
        # Store in database for analysis
        pass

    def analyze_results(self):
        """Compare performance of model A vs B"""
        # Calculate accuracy, user satisfaction for each model
        pass
```

---

### Step 6: Automated Retraining Scheduler

**Objective**: Automatically retrain model when enough feedback is collected

**Implementation**:

```python
# src/retraining_scheduler.py

import schedule
import time
from datetime import datetime

class RetrainingScheduler:
    def __init__(self, min_feedback_threshold=100):
        self.min_feedback_threshold = min_feedback_threshold
        self.trainer = SentimentModelTrainer()

    def check_and_retrain(self):
        """Check if retraining is needed and execute"""
        db = FeedbackDatabase()
        feedback_count = db.get_feedback_count()

        if feedback_count >= self.min_feedback_threshold:
            print(f"[{datetime.now()}] Starting model retraining...")
            print(f"Feedback samples: {feedback_count}")

            # Retrain model
            new_model_dir = self.trainer.retrain_with_feedback()

            if new_model_dir:
                # Evaluate new model
                self.evaluate_and_deploy(new_model_dir)
            else:
                print("Retraining failed or insufficient data")

    def evaluate_and_deploy(self, model_dir):
        """Evaluate fine-tuned model and deploy if better"""
        # Run evaluation on test set
        # Compare with current model
        # If better, deploy new model
        # Update database with new model version
        pass

    def start_scheduler(self):
        """Start automated retraining schedule"""
        # Check every day at 2 AM
        schedule.every().day.at("02:00").do(self.check_and_retrain)

        while True:
            schedule.run_pending()
            time.sleep(3600)  # Check every hour
```

---

### Step 7: Performance Monitoring Dashboard

**Objective**: Visualize model performance over time

**Implementation**:

```python
# Add to app.py - Admin dashboard section

def show_admin_dashboard():
    """Display model performance metrics"""
    st.title("Model Performance Dashboard")

    db = FeedbackDatabase()

    # Feedback statistics
    total_feedback = db.get_feedback_count()
    correct_predictions = db.get_correct_count()
    incorrect_predictions = db.get_incorrect_count()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Feedback", total_feedback)
    with col2:
        st.metric("Correct", correct_predictions)
    with col3:
        st.metric("Incorrect", incorrect_predictions)

    # Accuracy over time
    st.subheader("Accuracy Trend")
    accuracy_data = db.get_accuracy_over_time()
    fig = px.line(accuracy_data, x='date', y='accuracy',
                 title='Model Accuracy Over Time')
    st.plotly_chart(fig)

    # Confusion matrix
    st.subheader("Confusion Matrix")
    confusion_data = db.get_confusion_matrix()
    # Visualize confusion matrix

    # Recent feedback
    st.subheader("Recent Feedback")
    recent = db.get_recent_feedback(limit=10)
    st.dataframe(recent)
```

---

### Step 8: Deployment and Rollback Strategy

**Objective**: Safely deploy updated models with ability to rollback

**Implementation**:

```python
# src/model_deployer.py

import shutil
from pathlib import Path

class ModelDeployer:
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.current_model = self.models_dir / "current"
        self.backup_model = self.models_dir / "backup"

    def deploy_new_model(self, new_model_dir):
        """Deploy new model with backup"""

        # Backup current model
        if self.current_model.exists():
            if self.backup_model.exists():
                shutil.rmtree(self.backup_model)
            shutil.copytree(self.current_model, self.backup_model)

        # Deploy new model
        if self.current_model.exists():
            shutil.rmtree(self.current_model)
        shutil.copytree(new_model_dir, self.current_model)

        print(f"New model deployed: {new_model_dir}")
        print("Backup created at: {self.backup_model}")

    def rollback(self):
        """Rollback to previous model"""
        if not self.backup_model.exists():
            raise Exception("No backup model available")

        if self.current_model.exists():
            shutil.rmtree(self.current_model)
        shutil.copytree(self.backup_model, self.current_model)

        print("Rolled back to previous model")
```

---

## Technical Specifications

### Dependencies

Add to `requirements.txt`:
```
# Existing dependencies
streamlit==1.28.1
transformers==4.36.2
torch==2.1.2

# New dependencies
datasets==2.14.5
schedule==1.2.0
scikit-learn==1.3.2
```

### File Structure After Enhancement

```
nlpapplication2assignment/
├── src/
│   ├── sentiment_analyzer.py (updated)
│   ├── database.py (new)
│   ├── feedback_validator.py (new)
│   ├── model_trainer.py (new)
│   ├── ab_testing.py (new)
│   ├── retraining_scheduler.py (new)
│   └── model_deployer.py (new)
├── data/
│   └── feedback.db (created at runtime)
├── models/
│   ├── current/ (active model)
│   ├── backup/ (previous model)
│   └── fine_tuned/ (retrained models)
└── app.py (updated with feedback UI)
```

---

## Model Refinement Strategy

### Active Learning Workflow

```
1. User provides feedback on prediction
   ↓
2. Feedback stored in database
   ↓
3. When threshold reached (e.g., 100 samples):
   ↓
4. Data validation and quality filtering
   ↓
5. Fine-tune model on validated feedback
   ↓
6. Evaluate on held-out test set
   ↓
7. Compare with current model performance
   ↓
8. If improvement > 1%: Deploy new model
   ↓
9. Monitor performance via A/B testing
   ↓
10. Repeat cycle
```

### Fine-Tuning Configuration

```python
training_config = {
    "learning_rate": 2e-5,
    "epochs": 3,
    "batch_size": 8,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "max_length": 512,
}
```

### Data Augmentation

For small feedback datasets, augment with:
- Synonym replacement
- Back-translation
- Paraphrasing (using T5)

---

## Evaluation Metrics

### Model Performance Metrics

1. **Accuracy**: Overall correct predictions
2. **Precision**: Per-class precision
3. **Recall**: Per-class recall
4. **F1-Score**: Harmonic mean
5. **Confusion Matrix**: Detailed error analysis

### User Engagement Metrics

1. **Feedback Rate**: % of users providing feedback
2. **Correction Rate**: % of incorrect predictions corrected
3. **User Satisfaction**: Based on positive feedback ratio
4. **Session Duration**: Time spent using app

### Model Evolution Metrics

1. **Accuracy Improvement**: % increase from baseline
2. **Error Reduction**: Decrease in misclassifications
3. **Confidence Calibration**: Alignment of confidence with accuracy
4. **Domain Adaptation**: Performance on specialized vocabulary

---

## Deployment Plan

### Phase 1: Beta Release (Week 1-2)
- Deploy feedback collection UI
- Limited user testing (10-20 users)
- Monitor database performance

### Phase 2: Data Collection (Week 3-6)
- Full deployment of feedback system
- Aim for 500+ feedback samples
- Quality monitoring

### Phase 3: First Retraining (Week 7)
- Fine-tune model on collected data
- Evaluate on test set
- A/B test with original model

### Phase 4: Production Rollout (Week 8+)
- Deploy improved model if performance gain
- Enable automated retraining
- Continuous monitoring

---

## Timeline and Resources

### Implementation Timeline

| Phase | Duration | Tasks |
|-------|----------|-------|
| Phase 1: Database Setup | 1 week | Create schema, implement CRUD operations |
| Phase 2: UI Integration | 1 week | Add feedback buttons, forms |
| Phase 3: Validation System | 1 week | Implement quality checks |
| Phase 4: Training Pipeline | 2 weeks | Build fine-tuning pipeline |
| Phase 5: A/B Testing | 1 week | Implement testing framework |
| Phase 6: Automation | 1 week | Scheduler and deployment |
| Phase 7: Dashboard | 1 week | Monitoring and analytics |
| Phase 8: Testing & Deployment | 2 weeks | End-to-end testing |

**Total Duration**: 10 weeks

### Required Resources

**Personnel**:
- 1 ML Engineer (model training)
- 1 Backend Developer (database, pipeline)
- 1 Frontend Developer (UI enhancements)
- 1 QA Tester

**Infrastructure**:
- Database server (or SQLite for MVP)
- Model training environment (GPU recommended)
- Monitoring system

**Budget Estimate**: ~₹50,000-100,000 for cloud resources (if using cloud GPU)

---

## Conclusion

This enhancement plan provides a comprehensive roadmap for implementing a real-time feedback system that will continuously improve the sentiment analysis model. The system is designed to be scalable, maintainable, and measurable.

### Key Benefits:
- **Improved Accuracy**: Expected 2-5% improvement
- **User Engagement**: Interactive feedback increases user satisfaction
- **Continuous Learning**: Model adapts to new data and domains
- **Measurable Impact**: Clear metrics for tracking improvement

### Next Steps:
1. Approve enhancement plan
2. Allocate resources
3. Begin Phase 1 implementation
4. Iterate based on user feedback

---

**Document Prepared By**: BITS Pilani M.Tech AIML Student
**Date**: February 2026
**Assignment**: NLP Applications Assignment 2 - PS-9 (Task B)
