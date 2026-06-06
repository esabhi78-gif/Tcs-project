"""
Text Preprocessing Utilities for the Smart Ticket Understanding Engine.
Handles cleaning, normalization, and tokenization of ticket text.
"""

import re
import string


def clean_text(text):
    """
    Clean and normalize ticket text for ML processing.
    
    Steps:
      1. Lowercase conversion
      2. Remove email addresses
      3. Remove URLs
      4. Remove ticket/reference numbers
      5. Remove excessive punctuation but keep sentiment-carrying ones
      6. Normalize whitespace
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove ticket/reference numbers (e.g., INC00012345, TKT-2024-001)
    text = re.sub(r'[a-z]{2,5}[-]?\d{4,}', '', text)

    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-z0-9\s\.\,\!\?\'\-]', ' ', text)

    # Remove excessive punctuation (e.g., "!!!" → "!")
    text = re.sub(r'([!?.])\1+', r'\1', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_features(text):
    """
    Extract hand-crafted features from ticket text to supplement TF-IDF.
    
    Returns a dict of features:
      - text_length: character count
      - word_count: number of words
      - has_urgent_keywords: 1/0 
      - has_error_keywords: 1/0
      - exclamation_count: number of '!'
      - question_count: number of '?'
      - all_caps_ratio: ratio of uppercase words in original text
    """
    features = {}

    # Basic counts
    features['text_length'] = len(text)
    features['word_count'] = len(text.split())

    text_lower = text.lower()

    # Urgency indicators
    urgent_words = {'urgent', 'asap', 'critical', 'emergency', 'immediately',
                    'blocking', 'down', 'outage', 'crash', 'broken', 'unable',
                    'cannot', "can't", 'failing', 'failed'}
    features['has_urgent_keywords'] = int(
        any(w in text_lower for w in urgent_words)
    )

    # Error/issue indicators
    error_words = {'error', 'exception', 'timeout', 'denied', '500', '404',
                   'refused', 'rejected', 'expired', 'invalid', 'unauthorized'}
    features['has_error_keywords'] = int(
        any(w in text_lower for w in error_words)
    )

    # Punctuation-based sentiment signals
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')

    # Caps ratio (from original text, before cleaning)
    words = text.split()
    if words:
        caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        features['all_caps_ratio'] = round(caps_words / len(words), 3)
    else:
        features['all_caps_ratio'] = 0.0

    return features
