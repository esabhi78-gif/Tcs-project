"""
Smart Ticket Understanding Engine — Prediction Pipeline
Loads trained models and provides a unified prediction interface.
"""

import os
import sys
import joblib
import json
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.preprocessing import clean_text
from utils.actions import get_recommended_action, get_department, get_sla_hours

# Try to import VADER for sentiment analysis
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False


class TicketPredictor:
    """
    Unified prediction engine for IT service desk tickets.
    
    Loads trained scikit-learn models and provides:
      - Category (Incident Type) classification
      - Priority classification
      - Department routing
      - Sentiment analysis (VADER-based)
      - Recommended action generation
    """

    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = os.path.join(PROJECT_ROOT, "models", "saved")

        self.models_dir = models_dir
        self.models = {}
        self.meta = {}
        self._load_models()

        # Initialize sentiment analyzer
        if VADER_AVAILABLE:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        else:
            self.sentiment_analyzer = None

    def _load_models(self):
        """Load all saved models from disk."""
        targets = ['category', 'priority', 'department']
        
        for target in targets:
            model_path = os.path.join(self.models_dir, f"{target}_model.joblib")
            if os.path.exists(model_path):
                self.models[target] = joblib.load(model_path)
            else:
                print(f"⚠️  Model not found: {model_path}")

        # Load training metadata
        meta_path = os.path.join(self.models_dir, "training_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.meta = json.load(f)

    def is_ready(self):
        """Check if all required models are loaded."""
        required = ['category', 'priority', 'department']
        return all(t in self.models for t in required)

    def analyze_sentiment(self, text):
        """
        Analyze sentiment of ticket text using VADER.
        
        Maps VADER compound score to ticket-relevant sentiment labels:
          - compound >= 0.3  → Positive
          - compound <= -0.5 → Frustrated
          - has urgency keywords → Urgent
          - else → Neutral
        """
        if self.sentiment_analyzer is None:
            return self._fallback_sentiment(text)

        scores = self.sentiment_analyzer.polarity_scores(text)
        compound = scores['compound']

        # Check for urgency keywords first (overrides sentiment)
        urgency_words = {
            'urgent', 'asap', 'immediately', 'emergency', 'critical',
            'blocking', 'can\'t work', 'unable to work', 'down',
            'outage', 'production down', 'system down',
        }
        text_lower = text.lower()
        if any(w in text_lower for w in urgency_words):
            return "Urgent", scores

        if compound >= 0.3:
            return "Positive", scores
        elif compound <= -0.5:
            return "Frustrated", scores
        else:
            return "Neutral", scores

    def _fallback_sentiment(self, text):
        """Simple keyword-based sentiment fallback when VADER is unavailable."""
        text_lower = text.lower()
        
        urgency = {'urgent', 'asap', 'immediately', 'emergency', 'critical', 'blocking'}
        frustration = {'frustrated', 'disappointed', 'unacceptable', 'terrible', 'angry', 'worst'}
        positive = {'thanks', 'great', 'appreciate', 'helpful', 'excellent', 'love'}

        if any(w in text_lower for w in urgency):
            return "Urgent", {"compound": -0.8}
        if any(w in text_lower for w in frustration):
            return "Frustrated", {"compound": -0.6}
        if any(w in text_lower for w in positive):
            return "Positive", {"compound": 0.5}
        return "Neutral", {"compound": 0.0}

    def predict(self, ticket_text):
        """
        Predict all attributes for a single ticket.
        
        Args:
            ticket_text: Raw ticket text string
            
        Returns:
            dict with keys: category, priority, department, sentiment,
                           sentiment_scores, recommended_action, sla_hours,
                           confidence
        """
        if not self.is_ready():
            raise RuntimeError("Models not loaded. Run training first.")

        # Clean text for ML models
        cleaned = clean_text(ticket_text)

        # Predict category, priority, department
        category = self.models['category'].predict([cleaned])[0]
        priority = self.models['priority'].predict([cleaned])[0]
        department = self.models['department'].predict([cleaned])[0]

        # Get confidence scores where available
        confidence = {}
        for target in ['category', 'priority', 'department']:
            model = self.models[target]
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba([cleaned])[0]
                confidence[target] = round(float(np.max(proba)), 4)
            elif hasattr(model.named_steps.get('clf', None), 'decision_function'):
                # For LinearSVC, use decision function as proxy
                try:
                    dec = model.decision_function([cleaned])[0]
                    if isinstance(dec, np.ndarray):
                        confidence[target] = round(float(np.max(dec)), 4)
                    else:
                        confidence[target] = round(float(abs(dec)), 4)
                except Exception:
                    confidence[target] = None
            else:
                confidence[target] = None

        # Sentiment analysis (independent of ML models)
        sentiment, sentiment_scores = self.analyze_sentiment(ticket_text)

        # Get recommended action
        action = get_recommended_action(category, priority, sentiment)
        sla = get_sla_hours(priority)

        return {
            "ticket_text": ticket_text,
            "category": category,
            "priority": priority,
            "department": department,
            "sentiment": sentiment,
            "sentiment_scores": sentiment_scores,
            "recommended_action": action,
            "sla_hours": sla,
            "confidence": confidence,
        }

    def predict_batch(self, tickets):
        """
        Predict attributes for multiple tickets.
        
        Args:
            tickets: List of ticket text strings
            
        Returns:
            List of prediction dicts
        """
        return [self.predict(t) for t in tickets]

    def get_model_info(self):
        """Return information about loaded models."""
        info = {
            "models_loaded": list(self.models.keys()),
            "all_models_ready": self.is_ready(),
            "vader_available": VADER_AVAILABLE,
            "training_meta": self.meta,
        }
        return info
