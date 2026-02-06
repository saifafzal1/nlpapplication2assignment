"""
Visualization Module
This module provides visualization functions for sentiment analysis results.
Creates interactive charts and color-coded displays using Plotly.
"""

import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import pandas as pd

# Import configuration for colors
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COLOR_POSITIVE, COLOR_NEGATIVE, COLOR_NEUTRAL


def get_sentiment_color(sentiment: str) -> str:
    """
    Get the color associated with a sentiment label.

    Args:
        sentiment: Sentiment label ('POSITIVE', 'NEGATIVE', or 'NEUTRAL')

    Returns:
        Hex color code string

    Example:
        >>> get_sentiment_color('POSITIVE')
        '#28a745'
    """
    color_map = {
        'POSITIVE': COLOR_POSITIVE,
        'NEGATIVE': COLOR_NEGATIVE,
        'NEUTRAL': COLOR_NEUTRAL
    }
    return color_map.get(sentiment.upper(), '#6c757d')


def create_sentiment_bar_chart(sentiment_scores: Dict[str, float]) -> go.Figure:
    """
    Create a horizontal bar chart showing sentiment scores.

    This function creates an interactive Plotly bar chart that displays
    the confidence scores for each sentiment class (positive, negative, neutral).
    Each bar is color-coded according to the sentiment.

    Args:
        sentiment_scores: Dictionary mapping sentiment labels to scores
                         Example: {'POSITIVE': 0.95, 'NEGATIVE': 0.03, 'NEUTRAL': 0.02}

    Returns:
        Plotly Figure object containing the bar chart

    Example:
        >>> scores = {'POSITIVE': 0.85, 'NEGATIVE': 0.10, 'NEUTRAL': 0.05}
        >>> fig = create_sentiment_bar_chart(scores)
        >>> fig.show()
    """
    # Prepare data
    sentiments = list(sentiment_scores.keys())
    scores = list(sentiment_scores.values())
    colors = [get_sentiment_color(s) for s in sentiments]

    # Create horizontal bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=scores,
            y=sentiments,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='white', width=1)
            ),
            text=[f"{score:.2%}" for score in scores],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Confidence: %{x:.2%}<extra></extra>'
        )
    ])

    # Update layout
    fig.update_layout(
        title={
            'text': 'Sentiment Confidence Scores',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#333333'}
        },
        xaxis_title='Confidence Score',
        yaxis_title='Sentiment',
        xaxis=dict(
            tickformat='.0%',
            range=[0, 1],
            gridcolor='lightgray'
        ),
        yaxis=dict(
            categoryorder='total ascending'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        font=dict(size=12)
    )

    return fig


def create_sentiment_gauge(sentiment_score: float, sentiment_label: str) -> go.Figure:
    """
    Create a gauge chart showing the confidence level for the predicted sentiment.

    Args:
        sentiment_score: Confidence score (0.0 to 1.0)
        sentiment_label: Sentiment label string

    Returns:
        Plotly Figure object containing the gauge chart

    Example:
        >>> fig = create_sentiment_gauge(0.95, 'POSITIVE')
        >>> fig.show()
    """
    color = get_sentiment_color(sentiment_label)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=sentiment_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{sentiment_label} Confidence", 'font': {'size': 16}},
        number={'suffix': "%", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': '#ffebee'},
                {'range': [33, 66], 'color': '#fff9e6'},
                {'range': [66, 100], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='white',
        font={'size': 12}
    )

    return fig


def create_sentiment_pie_chart(sentiment_scores: Dict[str, float]) -> go.Figure:
    """
    Create a pie chart showing the distribution of sentiment scores.

    Args:
        sentiment_scores: Dictionary mapping sentiment labels to scores

    Returns:
        Plotly Figure object containing the pie chart

    Example:
        >>> scores = {'POSITIVE': 0.85, 'NEGATIVE': 0.10, 'NEUTRAL': 0.05}
        >>> fig = create_sentiment_pie_chart(scores)
        >>> fig.show()
    """
    sentiments = list(sentiment_scores.keys())
    scores = list(sentiment_scores.values())
    colors = [get_sentiment_color(s) for s in sentiments]

    fig = go.Figure(data=[go.Pie(
        labels=sentiments,
        values=scores,
        hole=0.3,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        textinfo='label+percent',
        textposition='auto',
        hovertemplate='<b>%{label}</b><br>Score: %{value:.2%}<extra></extra>'
    )])

    fig.update_layout(
        title={
            'text': 'Sentiment Distribution',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    return fig


def get_sentiment_color_box_style(sentiment: str) -> str:
    """
    Get CSS-like styling for a colored sentiment box.

    This function returns a style string that can be used with Streamlit's
    markdown component to create color-coded sentiment displays.

    Args:
        sentiment: Sentiment label string

    Returns:
        CSS style string

    Example:
        >>> style = get_sentiment_color_box_style('POSITIVE')
        >>> # Use with Streamlit: st.markdown(f'<div style="{style}">POSITIVE</div>', ...)
    """
    color = get_sentiment_color(sentiment)

    # Calculate lighter background color (add transparency)
    bg_color = f"{color}20"  # 20 is hex for ~12% opacity

    style = f"""
        background-color: {bg_color};
        color: {color};
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid {color};
        font-weight: bold;
        font-size: 18px;
        text-align: center;
        margin: 10px 0;
    """

    return style


def create_preprocessing_comparison_html(
    original: str,
    processed: str,
    steps: List[str]
) -> str:
    """
    Create an HTML comparison view of original vs preprocessed text.

    Args:
        original: Original input text
        processed: Preprocessed text
        steps: List of preprocessing steps applied

    Returns:
        HTML string for rendering the comparison

    Example:
        >>> html = create_preprocessing_comparison_html(
        ...     "I'm loving this!",
        ...     "love",
        ...     ['cleaning', 'tokenization', 'lemmatization']
        ... )
    """
    steps_str = " → ".join(steps)

    html = f"""
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin: 10px 0;">
        <h4 style="color: #333; margin-bottom: 15px;">Preprocessing Comparison</h4>

        <div style="margin-bottom: 15px;">
            <p style="color: #666; font-size: 12px; margin-bottom: 5px;">
                <b>Steps Applied:</b> {steps_str}
            </p>
        </div>

        <div style="background-color: white; padding: 15px; border-radius: 5px; margin-bottom: 10px;">
            <p style="color: #666; font-size: 12px; margin: 0; margin-bottom: 5px;">
                <b>Original Text:</b>
            </p>
            <p style="color: #333; margin: 0; font-family: monospace;">
                {original}
            </p>
        </div>

        <div style="background-color: white; padding: 15px; border-radius: 5px;">
            <p style="color: #666; font-size: 12px; margin: 0; margin-bottom: 5px;">
                <b>Preprocessed Text:</b>
            </p>
            <p style="color: #333; margin: 0; font-family: monospace;">
                {processed}
            </p>
        </div>
    </div>
    """

    return html


def create_sentiment_summary_table(results: List[Dict]) -> pd.DataFrame:
    """
    Create a summary table from multiple sentiment analysis results.

    Useful for batch analysis or displaying historical results.

    Args:
        results: List of sentiment analysis result dictionaries

    Returns:
        Pandas DataFrame with formatted results

    Example:
        >>> results = [
        ...     {'text': 'I love this', 'label': 'POSITIVE', 'score': 0.95},
        ...     {'text': 'This is bad', 'label': 'NEGATIVE', 'score': 0.89}
        ... ]
        >>> df = create_sentiment_summary_table(results)
    """
    data = []
    for result in results:
        data.append({
            'Text': result.get('text', '')[:50] + '...' if len(result.get('text', '')) > 50 else result.get('text', ''),
            'Sentiment': result.get('label', 'N/A'),
            'Confidence': f"{result.get('score', 0):.2%}"
        })

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    # Example usage and testing
    print("Visualization Module Test")
    print("=" * 50)

    # Test sentiment scores
    test_scores = {
        'POSITIVE': 0.85,
        'NEGATIVE': 0.10,
        'NEUTRAL': 0.05
    }

    print("\nCreating bar chart...")
    fig1 = create_sentiment_bar_chart(test_scores)
    print("Bar chart created successfully!")

    print("\nCreating gauge chart...")
    fig2 = create_sentiment_gauge(0.85, 'POSITIVE')
    print("Gauge chart created successfully!")

    print("\nCreating pie chart...")
    fig3 = create_sentiment_pie_chart(test_scores)
    print("Pie chart created successfully!")

    print("\nTesting color functions...")
    print(f"Positive color: {get_sentiment_color('POSITIVE')}")
    print(f"Negative color: {get_sentiment_color('NEGATIVE')}")
    print(f"Neutral color: {get_sentiment_color('NEUTRAL')}")

    print("\nCreating preprocessing comparison HTML...")
    html = create_preprocessing_comparison_html(
        "I'm absolutely loving this amazing product!!!",
        "absolutely love amazing product",
        ['cleaning', 'tokenization', 'stopword_removal', 'lemmatization']
    )
    print("HTML created successfully!")
    print(f"HTML length: {len(html)} characters")

    print("\n" + "=" * 50)
    print("All visualization tests passed!")
