"""
Sentiment Analysis Module
This module provides sentiment analysis functionality using Hugging Face transformers.
Uses pre-trained DistilBERT model for accurate sentiment prediction.
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_NAME, MAX_TEXT_LENGTH, CONFIDENCE_THRESHOLD


class SentimentAnalyzer:
    """
    A class for performing sentiment analysis using transformer models.

    This class uses Hugging Face's transformers library to load pre-trained
    models (default: DistilBERT) and perform sentiment classification on text.
    It provides both simple and detailed sentiment analysis with confidence scores.
    """

    def __init__(self, model_name: str = None):
        """
        Initialize the sentiment analyzer with a pre-trained model.

        Args:
            model_name: Name of the Hugging Face model to use.
                       Defaults to config.MODEL_NAME if not specified.

        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> # or specify a different model
            >>> analyzer = SentimentAnalyzer("cardiffnlp/twitter-roberta-base-sentiment")
        """
        self.model_name = model_name or MODEL_NAME
        self.max_length = MAX_TEXT_LENGTH
        self.threshold = CONFIDENCE_THRESHOLD
        self.classifier = None
        self.tokenizer = None
        self.model = None

        # Load the model
        self._load_model()

    def _load_model(self):
        """
        Load the pre-trained sentiment analysis model and tokenizer.

        This method initializes the transformer pipeline for sentiment analysis.
        The model is cached locally after first download for faster subsequent loads.

        Raises:
            Exception: If model loading fails
        """
        try:
            print(f"Loading model: {self.model_name}...")

            # Load tokenizer and model separately for more control
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )

            # Create pipeline
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1  # Use CPU (-1), change to 0 for GPU
            )

            print(f"Model loaded successfully!")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def analyze(self, text: str) -> Dict[str, any]:
        """
        Analyze the sentiment of the given text.

        This method performs sentiment analysis and returns the predicted label,
        confidence score, and additional metadata.

        Args:
            text: The input text string to analyze

        Returns:
            Dictionary containing:
                - label: Sentiment label ('POSITIVE', 'NEGATIVE', or 'NEUTRAL')
                - score: Confidence score (0.0 to 1.0)
                - confidence_percent: Score as percentage string
                - all_scores: Dictionary with scores for all sentiment classes

        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> result = analyzer.analyze("I love this product!")
            >>> print(result['label'])
            'POSITIVE'
            >>> print(result['score'])
            0.9998
        """
        if not text or text.strip() == "":
            return {
                'label': 'NEUTRAL',
                'score': 0.0,
                'confidence_percent': '0.00%',
                'all_scores': {'POSITIVE': 0.0, 'NEGATIVE': 0.0, 'NEUTRAL': 1.0}
            }

        try:
            # Truncate text if too long
            if len(text.split()) > self.max_length:
                tokens = text.split()[:self.max_length]
                text = ' '.join(tokens)

            # Run sentiment analysis
            results = self.classifier(text)[0]

            # Extract label and score
            label = results['label'].upper()
            score = results['score']

            # Get all sentiment scores
            all_scores = self._get_all_scores(text)

            # For binary models (POSITIVE/NEGATIVE only), determine NEUTRAL
            if label in ['POSITIVE', 'NEGATIVE']:
                # If confidence is below threshold, classify as neutral
                if score < self.threshold:
                    label = 'NEUTRAL'

            return {
                'label': label,
                'score': score,
                'confidence_percent': f"{score * 100:.2f}%",
                'all_scores': all_scores
            }

        except Exception as e:
            print(f"Error during sentiment analysis: {e}")
            return {
                'label': 'ERROR',
                'score': 0.0,
                'confidence_percent': '0.00%',
                'all_scores': {},
                'error': str(e)
            }

    def _get_all_scores(self, text: str) -> Dict[str, float]:
        """
        Get confidence scores for all sentiment classes.

        This method uses the model's raw output to extract probability
        scores for all possible sentiment labels.

        Args:
            text: The input text string

        Returns:
            Dictionary mapping sentiment labels to confidence scores

        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> scores = analyzer._get_all_scores("This is okay")
            >>> print(scores)
            {'POSITIVE': 0.45, 'NEGATIVE': 0.55}
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            )

            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Get probabilities using softmax
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            probs = probabilities[0].tolist()

            # Map probabilities to labels
            label_mapping = self.model.config.id2label

            scores = {}
            for idx, prob in enumerate(probs):
                label = label_mapping[idx].upper()
                # Normalize label names
                if 'POS' in label:
                    scores['POSITIVE'] = prob
                elif 'NEG' in label:
                    scores['NEGATIVE'] = prob
                elif 'NEU' in label:
                    scores['NEUTRAL'] = prob
                else:
                    scores[label] = prob

            # For binary models, calculate neutral as average
            if 'NEUTRAL' not in scores and 'POSITIVE' in scores and 'NEGATIVE' in scores:
                max_score = max(scores['POSITIVE'], scores['NEGATIVE'])
                if max_score < self.threshold:
                    scores['NEUTRAL'] = 1.0 - max_score
                else:
                    scores['NEUTRAL'] = 0.0

            return scores

        except Exception as e:
            print(f"Error getting all scores: {e}")
            return {'POSITIVE': 0.0, 'NEGATIVE': 0.0, 'NEUTRAL': 0.0}

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        Analyze sentiment for multiple texts in batch.

        Batch processing is more efficient than analyzing texts one by one
        when you have multiple texts to process.

        Args:
            texts: List of text strings to analyze

        Returns:
            List of dictionaries, each containing sentiment analysis results

        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> texts = ["I love this!", "I hate this!", "It's okay"]
            >>> results = analyzer.analyze_batch(texts)
            >>> for result in results:
            ...     print(result['label'])
            POSITIVE
            NEGATIVE
            NEUTRAL
        """
        return [self.analyze(text) for text in texts]

    def get_sentiment_emoji(self, label: str) -> str:
        """
        Get an emoji representation for a sentiment label.

        Args:
            label: Sentiment label string

        Returns:
            Emoji string representing the sentiment

        Note: This method is optional and not used by default (as per requirements)
        """
        emoji_map = {
            'POSITIVE': '😊',
            'NEGATIVE': '😞',
            'NEUTRAL': '😐'
        }
        return emoji_map.get(label.upper(), '❓')


def quick_sentiment_check(text: str) -> str:
    """
    Convenience function for quick sentiment analysis.

    This is a simple wrapper function that creates an analyzer instance
    and returns just the sentiment label.

    Args:
        text: The input text string

    Returns:
        Sentiment label string ('POSITIVE', 'NEGATIVE', or 'NEUTRAL')

    Example:
        >>> sentiment = quick_sentiment_check("This is amazing!")
        >>> print(sentiment)
        'POSITIVE'
    """
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze(text)
    return result['label']


if __name__ == "__main__":
    # Example usage and testing
    print("Sentiment Analysis Module Test")
    print("=" * 50)

    # Initialize analyzer
    analyzer = SentimentAnalyzer()

    # Test cases
    test_texts = [
        "I absolutely love this product! It's amazing and works perfectly!",
        "This is terrible. I'm very disappointed with the quality.",
        "It's okay, nothing special but not bad either.",
        "The service was good but the product could be better.",
        ""
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}:")
        print(f"Text: {text[:60]}{'...' if len(text) > 60 else ''}")

        result = analyzer.analyze(text)

        print(f"Sentiment: {result['label']}")
        print(f"Confidence: {result['confidence_percent']}")
        print(f"All Scores: {result['all_scores']}")
        print("-" * 50)
