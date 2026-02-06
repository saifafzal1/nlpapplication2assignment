"""
Sentiment Analysis Web Application
Built with Streamlit for BITS Pilani M.Tech AIML - NLP Applications Assignment

This application performs sentiment analysis on user-provided text using
modern transformer models (DistilBERT) and traditional NLP preprocessing techniques.
"""

import streamlit as st
import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.sentiment_analyzer import SentimentAnalyzer
from src.preprocessor import TextPreprocessor, preprocess_for_transformer
from src.visualizer import (
    create_sentiment_bar_chart,
    create_sentiment_pie_chart,
    get_sentiment_color,
    create_preprocessing_comparison_html
)
from config import PAGE_TITLE, PAGE_ICON, MAX_FILE_SIZE_MB

# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for better styling
def load_custom_css():
    """Apply custom CSS styling to the Streamlit app."""
    st.markdown("""
        <style>
        .main {
            padding: 0rem 1rem;
        }
        .stAlert {
            padding: 1rem;
            border-radius: 0.5rem;
        }
        h1 {
            color: #1f77b4;
            padding-bottom: 1rem;
        }
        .sentiment-box {
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            margin: 20px 0;
        }
        </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_sentiment_analyzer():
    """
    Load and cache the sentiment analyzer model.

    This function uses Streamlit's caching to avoid reloading
    the model on every interaction, improving performance.
    """
    return SentimentAnalyzer()


@st.cache_resource
def load_text_preprocessor():
    """Load and cache the text preprocessor."""
    return TextPreprocessor()


def display_sentiment_result(result, show_visualization=True):
    """
    Display the sentiment analysis result with color-coded box and charts.

    Args:
        result: Dictionary containing sentiment analysis results
        show_visualization: Whether to show visualizations (default: True)
    """
    label = result['label']
    score = result['score']
    all_scores = result['all_scores']

    # Color-coded sentiment display
    color = get_sentiment_color(label)

    # Create colored box for sentiment
    st.markdown(
        f"""
        <div style="background-color: {color}20;
                    border-left: 5px solid {color};
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;">
            <h2 style="color: {color}; margin: 0;">
                Sentiment: {label}
            </h2>
            <p style="color: #666; font-size: 18px; margin: 10px 0 0 0;">
                Confidence: {result['confidence_percent']}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if show_visualization and all_scores:
        # Create two columns for visualizations
        col1, col2 = st.columns(2)

        with col1:
            # Bar chart
            st.plotly_chart(
                create_sentiment_bar_chart(all_scores),
                use_container_width=True
            )

        with col2:
            # Pie chart
            st.plotly_chart(
                create_sentiment_pie_chart(all_scores),
                use_container_width=True
            )


def main():
    """Main application function."""

    # Load custom CSS
    load_custom_css()

    # Header
    st.title("Sentiment Analysis Application")
    st.markdown("""
        Analyze the sentiment of text using state-of-the-art NLP models.
        This application uses **DistilBERT** for accurate sentiment classification
        and provides traditional **NLTK preprocessing** visualization.
    """)

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Input method selection
    input_method = st.sidebar.radio(
        "Choose Input Method:",
        ["Text Input", "File Upload"],
        help="Select how you want to provide text for analysis"
    )

    # Preprocessing options
    st.sidebar.subheader("Preprocessing Options")
    st.sidebar.markdown("*Note: Modern transformers require minimal preprocessing*")

    show_preprocessing = st.sidebar.checkbox(
        "Show Preprocessing Steps",
        value=False,
        help="Display traditional NLTK preprocessing (for educational purposes)"
    )

    if show_preprocessing:
        apply_cleaning = st.sidebar.checkbox("Apply Text Cleaning", value=True)
        apply_tokenization = st.sidebar.checkbox("Apply Tokenization", value=True)
        apply_stopword_removal = st.sidebar.checkbox(
            "Remove Stopwords",
            value=False,
            help="Caution: Removes important words like 'not', 'never'"
        )
        apply_lemmatization = st.sidebar.checkbox("Apply Lemmatization", value=False)

    # Model information
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Information")
    st.sidebar.info("""
        **Model:** DistilBERT
        **Accuracy:** ~95%
        **Speed:** Fast (CPU-optimized)
        **Classes:** Positive, Negative, Neutral
    """)

    # Main content area
    st.markdown("---")

    # Initialize components
    try:
        with st.spinner("Loading sentiment analysis model..."):
            analyzer = load_sentiment_analyzer()
            preprocessor = load_text_preprocessor()

        st.success("Model loaded successfully!")

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

    # Input handling
    text_to_analyze = None

    if input_method == "Text Input":
        st.subheader("Enter Text for Analysis")

        text_to_analyze = st.text_area(
            "Type or paste your text here:",
            height=150,
            placeholder="Example: I absolutely love this product! It's amazing and works perfectly.",
            help="Enter any text you want to analyze for sentiment"
        )

        # Quick examples
        st.markdown("**Quick Examples:**")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Positive Example"):
                text_to_analyze = "I absolutely love this product! It's amazing and works perfectly. Highly recommended!"

        with col2:
            if st.button("Negative Example"):
                text_to_analyze = "This is terrible. I'm very disappointed with the quality and service. Would not recommend."

        with col3:
            if st.button("Neutral Example"):
                text_to_analyze = "The product is okay. It works as described but nothing exceptional."

    else:  # File Upload
        st.subheader("Upload Text File")

        uploaded_file = st.file_uploader(
            "Choose a .txt file",
            type=['txt'],
            help=f"Maximum file size: {MAX_FILE_SIZE_MB}MB"
        )

        if uploaded_file is not None:
            try:
                # Read file
                file_size_mb = uploaded_file.size / (1024 * 1024)

                if file_size_mb > MAX_FILE_SIZE_MB:
                    st.error(f"File too large! Maximum size is {MAX_FILE_SIZE_MB}MB")
                else:
                    # Try different encodings
                    try:
                        text_to_analyze = uploaded_file.read().decode('utf-8')
                    except UnicodeDecodeError:
                        text_to_analyze = uploaded_file.read().decode('latin-1')

                    st.success(f"File uploaded successfully! Size: {file_size_mb:.2f}MB")

                    # Show preview
                    with st.expander("View File Content"):
                        st.text(text_to_analyze[:500] + "..." if len(text_to_analyze) > 500 else text_to_analyze)

            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

    # Analysis button and results
    if text_to_analyze:
        st.markdown("---")

        # Analyze button
        if st.button("Analyze Sentiment", type="primary"):

            if not text_to_analyze.strip():
                st.warning("Please enter some text to analyze.")
            else:
                # Show preprocessing if enabled
                if show_preprocessing:
                    st.subheader("Preprocessing Steps")

                    with st.spinner("Preprocessing text..."):
                        preprocess_result = preprocessor.preprocess_pipeline(
                            text_to_analyze,
                            apply_cleaning=apply_cleaning,
                            apply_tokenization=apply_tokenization,
                            apply_stopword_removal=apply_stopword_removal,
                            apply_lemmatization=apply_lemmatization
                        )

                    # Display preprocessing comparison
                    html_comparison = create_preprocessing_comparison_html(
                        preprocess_result['original'],
                        preprocess_result['processed_text'],
                        preprocess_result['steps_applied']
                    )
                    st.markdown(html_comparison, unsafe_allow_html=True)

                    # Show token count
                    st.info(f"**Tokens:** {len(preprocess_result.get('tokens', []))} → "
                           f"{len(preprocess_result['processed_tokens'])} (after preprocessing)")

                # Perform sentiment analysis
                st.subheader("Sentiment Analysis Results")

                with st.spinner("Analyzing sentiment..."):
                    # Use minimal preprocessing for transformer
                    clean_text = preprocess_for_transformer(text_to_analyze)
                    result = analyzer.analyze(clean_text)

                # Display results
                if 'error' in result:
                    st.error(f"Analysis error: {result['error']}")
                else:
                    display_sentiment_result(result, show_visualization=True)

                    # Additional metrics
                    st.markdown("### Detailed Scores")
                    cols = st.columns(len(result['all_scores']))
                    for i, (sentiment, score) in enumerate(result['all_scores'].items()):
                        with cols[i]:
                            st.metric(
                                label=sentiment,
                                value=f"{score:.2%}",
                                delta=None
                            )

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>Sentiment Analysis Application | BITS Pilani M.Tech AIML</p>
            <p>Assignment 2 - PS-9 | NLP Applications</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
