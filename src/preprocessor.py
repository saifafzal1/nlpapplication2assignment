"""
Text Preprocessing Module
This module provides functions for preprocessing text data using NLTK.
Includes tokenization, stopword removal, stemming, and lemmatization.
"""

import re
import nltk
from typing import Dict, List, Tuple
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data on first import
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)


class TextPreprocessor:
    """
    A class for preprocessing text data using various NLP techniques.

    This class provides methods for cleaning, tokenizing, and normalizing text
    through stemming and lemmatization. It demonstrates traditional NLP preprocessing
    techniques commonly used before applying sentiment analysis models.
    """

    def __init__(self):
        """Initialize the preprocessor with necessary NLTK tools."""
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text: str) -> str:
        """
        Clean the input text by removing special characters and extra whitespace.

        Args:
            text: The input text string to clean

        Returns:
            Cleaned text string with normalized whitespace

        Example:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.clean_text("Hello!!!   World   ")
            'Hello World'
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into individual words using NLTK word_tokenize.

        Args:
            text: The input text string to tokenize

        Returns:
            List of token strings

        Example:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.tokenize("Hello, world!")
            ['Hello', ',', 'world', '!']
        """
        return word_tokenize(text)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove common English stopwords from the token list.

        Note: Be cautious when removing stopwords for sentiment analysis,
        as words like "not", "no", "never" carry important sentiment information.

        Args:
            tokens: List of token strings

        Returns:
            List of tokens with stopwords removed

        Example:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.remove_stopwords(['this', 'is', 'a', 'test'])
            ['test']
        """
        return [token for token in tokens if token.lower() not in self.stop_words]

    def stem(self, tokens: List[str]) -> List[str]:
        """
        Apply Porter stemming to reduce words to their root form.

        Stemming removes suffixes to get the word stem. It's faster but less
        accurate than lemmatization. Example: "running" -> "run", "better" -> "better"

        Args:
            tokens: List of token strings

        Returns:
            List of stemmed tokens

        Example:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.stem(['running', 'runner', 'runs'])
            ['run', 'runner', 'run']
        """
        return [self.stemmer.stem(token) for token in tokens]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Apply lemmatization to reduce words to their dictionary form.

        Lemmatization uses vocabulary and morphological analysis to return
        the base or dictionary form of a word. More accurate than stemming.
        Example: "running" -> "run", "better" -> "good"

        Args:
            tokens: List of token strings

        Returns:
            List of lemmatized tokens

        Example:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.lemmatize(['running', 'ran', 'runs'])
            ['running', 'ran', 'run']
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def preprocess_pipeline(
        self,
        text: str,
        apply_cleaning: bool = True,
        apply_tokenization: bool = True,
        apply_stopword_removal: bool = False,
        apply_stemming: bool = False,
        apply_lemmatization: bool = False
    ) -> Dict[str, any]:
        """
        Apply a complete preprocessing pipeline with configurable steps.

        This method allows you to selectively apply preprocessing steps and
        track the transformation at each stage. Useful for understanding the
        impact of each preprocessing technique.

        Args:
            text: The input text string to preprocess
            apply_cleaning: Whether to clean the text (default: True)
            apply_tokenization: Whether to tokenize (default: True)
            apply_stopword_removal: Whether to remove stopwords (default: False)
            apply_stemming: Whether to apply stemming (default: False)
            apply_lemmatization: Whether to apply lemmatization (default: False)

        Returns:
            Dictionary containing:
                - original: Original input text
                - cleaned: Cleaned text (if apply_cleaning=True)
                - tokens: Tokenized text (if apply_tokenization=True)
                - processed_tokens: Tokens after stopword/stem/lemma (if applicable)
                - processed_text: Final processed text joined back into string
                - steps_applied: List of preprocessing steps that were applied

        Example:
            >>> preprocessor = TextPreprocessor()
            >>> result = preprocessor.preprocess_pipeline(
            ...     "I'm loving this amazing product!",
            ...     apply_lemmatization=True
            ... )
            >>> print(result['processed_text'])
        """
        result = {
            'original': text,
            'steps_applied': []
        }

        current_text = text

        # Step 1: Clean text
        if apply_cleaning:
            current_text = self.clean_text(current_text)
            result['cleaned'] = current_text
            result['steps_applied'].append('cleaning')

        # Step 2: Tokenization
        if apply_tokenization:
            tokens = self.tokenize(current_text)
            result['tokens'] = tokens
            result['steps_applied'].append('tokenization')
        else:
            tokens = [current_text]

        # Step 3: Stopword removal
        if apply_stopword_removal:
            tokens = self.remove_stopwords(tokens)
            result['steps_applied'].append('stopword_removal')

        # Step 4: Stemming (mutually exclusive with lemmatization)
        if apply_stemming and not apply_lemmatization:
            tokens = self.stem(tokens)
            result['steps_applied'].append('stemming')

        # Step 5: Lemmatization
        if apply_lemmatization:
            tokens = self.lemmatize(tokens)
            result['steps_applied'].append('lemmatization')

        result['processed_tokens'] = tokens
        result['processed_text'] = ' '.join(tokens)

        return result

    def get_preprocessing_comparison(
        self,
        text: str
    ) -> Tuple[str, str, List[str]]:
        """
        Get a comparison between original and preprocessed text.

        This is a convenience method that applies a standard preprocessing
        pipeline and returns key comparisons for visualization.

        Args:
            text: The input text string

        Returns:
            Tuple of (original_text, processed_text, steps_applied)
        """
        result = self.preprocess_pipeline(
            text,
            apply_cleaning=True,
            apply_tokenization=True,
            apply_lemmatization=True
        )

        return (
            result['original'],
            result['processed_text'],
            result['steps_applied']
        )


def preprocess_for_transformer(text: str) -> str:
    """
    Minimal preprocessing specifically for transformer models (BERT, DistilBERT).

    Modern transformer models use subword tokenization (WordPiece, BPE) and
    are trained on raw text. Aggressive preprocessing can hurt performance.
    This function only applies minimal cleaning while preserving important
    features like punctuation and negation words.

    Args:
        text: The input text string

    Returns:
        Minimally processed text suitable for transformer models

    Example:
        >>> preprocess_for_transformer("I'm not happy with this!!!   ")
        "I'm not happy with this!"

    Note:
        This function preserves:
        - Contractions (I'm, don't, won't)
        - Negation words (not, never, no)
        - Punctuation (important for sentiment)
        - Case information (can be useful for some models)
    """
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


if __name__ == "__main__":
    # Example usage and testing
    preprocessor = TextPreprocessor()

    sample_text = "I'm absolutely loving this amazing product! It's fantastic!!!"

    print("Original Text:")
    print(sample_text)
    print("\n" + "="*50 + "\n")

    # Full preprocessing pipeline
    result = preprocessor.preprocess_pipeline(
        sample_text,
        apply_cleaning=True,
        apply_tokenization=True,
        apply_stopword_removal=True,
        apply_lemmatization=True
    )

    print("Preprocessing Steps Applied:", result['steps_applied'])
    print("\nOriginal:", result['original'])
    print("Cleaned:", result.get('cleaned', 'N/A'))
    print("Tokens:", result.get('tokens', 'N/A'))
    print("Processed Tokens:", result['processed_tokens'])
    print("Final Processed Text:", result['processed_text'])
    print("\n" + "="*50 + "\n")

    # Transformer preprocessing
    transformer_text = preprocess_for_transformer(sample_text)
    print("Transformer Preprocessing:")
    print(transformer_text)
