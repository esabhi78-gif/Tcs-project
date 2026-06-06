"""
Smart Ticket Understanding Engine — Gemini LLM Classifier
Uses Google Gemini API for classification with reasoning.
"""

import os
import json
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Load environment variables
load_dotenv()

class GeminiTicketClassifier:
    """
    LLM-powered ticket classifier that uses Gemini to provide
    category, priority, department, and reasoning.
    """
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.is_configured = False
        
        if GEMINI_AVAILABLE and self.api_key:
            self.client = genai.Client(api_key=self.api_key)
            self.model_name = 'gemma-4-31b-it'
            self.is_configured = True
            
        self.categories = [
            "Network/VPN", "Hardware", "Software/Application", 
            "Account/Access", "Email/Communication", "Database", 
            "Security", "General IT"
        ]
        
    def is_ready(self):
        return self.is_configured
        
    def classify_ticket(self, ticket_text):
        """
        Classify a ticket using Gemini and get reasoning.
        Returns a dict with classification and reasoning.
        """
        if not self.is_ready():
            raise RuntimeError("Gemini API is not configured. Please check your GOOGLE_API_KEY.")
            
        prompt = f"""
You are an expert IT Service Desk AI assistant.
Your task is to classify the following IT support ticket.

Categories available:
{', '.join(self.categories)}

Priorities available:
Critical, High, Medium, Low

Departments available:
Infra Team, Desktop Support, Application Support, IAM Team, Collaboration Team, DBA Team, Security Team, Service Desk

Ticket: "{ticket_text}"

Analyze the ticket and provide your response as a valid JSON object with EXACTLY the following keys:
- "category": The best fitting category from the list above. If the ticket doesn't fit ANY of these categories, use "Outlier".
- "priority": The appropriate priority level.
- "department": The department that should handle this.
- "reasoning": A 1-2 sentence explanation of WHY you chose this category and priority. Be specific about what in the text led to this decision.
- "is_outlier": boolean true if you chose "Outlier", false otherwise.

Ensure the output is ONLY valid JSON.
"""
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            response_text = response.text.strip()
            
            # Clean up markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]
            elif response_text.startswith("```"):
                response_text = response_text[3:-3]
                
            result = json.loads(response_text)
            
            # Map sentiment (fallback to a basic check for this object structure)
            sentiment = "Neutral"
            ticket_lower = ticket_text.lower()
            if any(w in ticket_lower for w in ["urgent", "asap", "emergency", "critical", "blocking", "down", "outage"]):
                sentiment = "Urgent"
            elif any(w in ticket_lower for w in ["frustrated", "disappointed", "unacceptable", "terrible", "angry"]):
                sentiment = "Frustrated"
            elif any(w in ticket_lower for w in ["thanks", "great", "appreciate"]):
                sentiment = "Positive"
                
            result["sentiment"] = sentiment
            
            return result
            
        except Exception as e:
            print(f"Gemini API Error: {e}")
            return None

    def classify_batch(self, tickets_list):
        """
        Classify multiple tickets in a SINGLE request to avoid rate limits.
        tickets_list: list of strings (max 20 recommended)
        """
        if not self.is_ready():
            return None
            
        tickets_text = ""
        for i, t in enumerate(tickets_list):
            tickets_text += f"Ticket {i}: \"{t}\"\n\n"
            
        prompt = f"""
You are an expert IT Service Desk AI assistant.
Your task is to classify multiple IT support tickets.

Categories available:
{', '.join(self.categories)}

Priorities available:
Critical, High, Medium, Low

Departments available:
Infra Team, Desktop Support, Application Support, IAM Team, Collaboration Team, DBA Team, Security Team, Service Desk

Here are the tickets to classify:
{tickets_text}

Analyze each ticket and provide your response as a valid JSON ARRAY of objects. 
Each object must have EXACTLY these keys:
- "index": the ticket index number
- "category": The best fitting category.
- "priority": The appropriate priority level.
- "department": The routing department.
- "reasoning": A 1 sentence explanation of why.
- "is_outlier": boolean

Ensure the output is ONLY a valid JSON array.
"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            response_text = response.text.strip()
            
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]
            elif response_text.startswith("```"):
                response_text = response_text[3:-3]
                
            results_array = json.loads(response_text)
            return results_array
            
        except Exception as e:
            print(f"Gemini API Batch Error: {e}")
            return None
