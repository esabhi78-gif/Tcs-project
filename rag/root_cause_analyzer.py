"""
Smart Ticket Understanding Engine — Root Cause Analyzer
Uses RAG (ChromaDB + Gemini) to find root causes and resolutions.
"""

import os
import json
from dotenv import load_dotenv

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from rag.vector_store import TicketVectorStore

load_dotenv()

class RootCauseAnalyzer:
    """
    RAG-based analyzer that retrieves similar tickets and uses
    Gemini to synthesize a likely root cause and resolution.
    """
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.is_configured = False
        
        if GEMINI_AVAILABLE and self.api_key:
            self.client = genai.Client(api_key=self.api_key)
            self.model_name = 'gemma-4-31b-it'
            self.is_configured = True
            
        self.vector_store = TicketVectorStore()
        
    def is_ready(self):
        return self.is_configured and self.vector_store.is_ready
        
    def analyze(self, ticket_text):
        """
        Analyze a ticket using RAG.
        Returns dict with similar tickets, root cause, and resolution.
        """
        # 1. Retrieve similar tickets
        similar = self.vector_store.search_similar(ticket_text, top_k=3)
        
        if not similar:
            return {
                "similar_tickets": [],
                "likely_root_cause": "Not enough historical data to determine.",
                "suggested_resolution": "Escalate to L2 support for investigation.",
                "confidence": "Low"
            }
            
        # Prepare context for LLM
        context_parts = []
        for i, s in enumerate(similar):
            context_parts.append(
                f"Historical Ticket {i+1}:\n"
                f"Description: {s['ticket_text']}\n"
                f"Root Cause: {s['metadata'].get('root_cause', 'Unknown')}\n"
                f"Resolution Steps: {s['metadata'].get('resolution_steps', 'Unknown')}\n"
            )
            
        context = "\n".join(context_parts)
        
        if not self.is_ready():
            # Fallback if Gemini not available
            return {
                "similar_tickets": similar,
                "likely_root_cause": similar[0]['metadata'].get('root_cause', 'Unknown') if similar else "Unknown",
                "suggested_resolution": similar[0]['metadata'].get('resolution_steps', 'Unknown') if similar else "Unknown",
                "confidence": "Based on closest match (LLM not configured)"
            }
            
        # 2. Ask Gemini to synthesize
        prompt = f"""
You are an expert IT Tier 3 Support Engineer. 
A new support ticket has been submitted. You need to analyze it and suggest a root cause and resolution steps.

To help you, here are {len(similar)} similar historical tickets that have already been resolved:

{context}

---
NEW TICKET: "{ticket_text}"
---

Based on the historical tickets and the new ticket description, provide:
1. The most likely root cause for the new ticket.
2. Step-by-step suggested resolution.
3. Your confidence level (High, Medium, Low) based on how well the historical tickets match the new one.

Provide your response as a valid JSON object with EXACTLY the following keys:
- "likely_root_cause": string
- "suggested_resolution": string
- "confidence": string

Ensure the output is ONLY valid JSON.
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
                
            result = json.loads(response_text)
            result["similar_tickets"] = similar
            
            return result
            
        except Exception as e:
            print(f"Gemini API Error in RAG: {e}")
            # Fallback to nearest neighbor
            return {
                "similar_tickets": similar,
                "likely_root_cause": similar[0]['metadata'].get('root_cause', 'Unknown') if similar else "Unknown",
                "suggested_resolution": similar[0]['metadata'].get('resolution_steps', 'Unknown') if similar else "Unknown",
                "confidence": "Error invoking LLM. Returning closest match."
            }
