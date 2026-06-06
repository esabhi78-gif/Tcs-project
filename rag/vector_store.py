"""
Smart Ticket Understanding Engine — Vector Store
Handles embedding and storing historical tickets in ChromaDB.
"""

import os
import pandas as pd

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TicketVectorStore:
    """Manages the ChromaDB vector database for RAG."""
    
    def __init__(self, db_path=None):
        if not CHROMA_AVAILABLE:
            self.is_ready = False
            return
            
        self.is_ready = True
        if db_path is None:
            db_path = os.path.join(PROJECT_ROOT, "rag", "chroma_db")
            
        # Ensure directory exists
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Use sentence-transformers (all-MiniLM-L6-v2 by default)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="historical_tickets",
            embedding_function=self.embedding_fn
        )
        
    def build_from_csv(self, csv_path):
        """Build the vector store from historical closed tickets."""
        if not self.is_ready:
            return False
            
        df = pd.read_csv(csv_path)
        
        # Only use closed tickets that have resolutions and root causes
        closed_tickets = df[df['ticket_status'] == 'Closed'].dropna(subset=['root_cause', 'resolution_steps'])
        
        if closed_tickets.empty:
            print("No closed tickets with resolutions found.")
            return False
            
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for idx, row in closed_tickets.iterrows():
            documents.append(str(row['ticket_text']))
            metadatas.append({
                "category": str(row['category']),
                "priority": str(row['priority']),
                "department": str(row['department']),
                "root_cause": str(row['root_cause']),
                "resolution_steps": str(row['resolution_steps'])
            })
            ids.append(str(row.get('ticket_id', f"tkt_{idx}")))
            
        # Clear existing collection and add new
        try:
            self.client.delete_collection("historical_tickets")
        except:
            pass
            
        self.collection = self.client.create_collection(
            name="historical_tickets",
            embedding_function=self.embedding_fn
        )
        
        # Add in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            self.collection.add(
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )
            
        print(f"✅ Added {len(documents)} tickets to vector store.")
        return True
        
    def search_similar(self, ticket_text, top_k=3):
        """Find similar historical tickets."""
        if not self.is_ready or self.collection.count() == 0:
            return []
            
        results = self.collection.query(
            query_texts=[ticket_text],
            n_results=top_k
        )
        
        similar_tickets = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                similar_tickets.append({
                    "ticket_text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results and results['distances'] else 0
                })
                
        return similar_tickets

if __name__ == "__main__":
    # Test script to build DB
    csv_path = os.path.join(PROJECT_ROOT, "data", "tickets.csv")
    if os.path.exists(csv_path):
        print(f"Building vector store from {csv_path}...")
        store = TicketVectorStore()
        store.build_from_csv(csv_path)
        print("Vector store ready.")
        
        # Test search
        res = store.search_similar("My laptop won't turn on, battery is dead.")
        print(f"\nFound {len(res)} similar tickets.")
        if res:
            print("Top match:", res[0]['ticket_text'])
    else:
        print("Dataset not found. Run generate_dataset.py first.")
