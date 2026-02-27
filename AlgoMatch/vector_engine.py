import numpy as np
from sentence_transformers import SentenceTransformer
from endee import Endee
from dsa_knowledge_base import dsa_patterns

class VectorEngine:
    def __init__(self, host="http://localhost:8080"):
        # Initialize Endee database client (it extracts base_url internal testing locally)
        # Endee SDK initializes with internal region localhost logic if no token is passed. 
        # But for localhost, we can set base_url directly using set_base_url
        self.client = Endee() 
        self.client.set_base_url(f"{host}/api/v1") 
        # Load HuggingFace embedder
        self.model = SentenceTransformer('all-MiniLM-L6-v2') 
        self.collection_name = "dsa_patterns"

    def initialize_db(self):
        """Creates the Endee vector collection and populates it with DSA patterns."""
        # Always recreate the index for this demo to avoid "required files missing" corruption 
        # from abandoned initializations
        try:
            self.client.delete_index(self.collection_name)
            print(f"Deleted existing/corrupted index {self.collection_name}")
        except Exception:
            pass # Index might not exist at all, which is fine
            
        try:
            # Create index with dimensions
            self.client.create_index(
                name=self.collection_name, 
                dimension=384, # 'all-MiniLM-L6-v2' outputs 384-dimensional embeddings
                space_type="cosine",
                M=16,
                ef_con=200,
                precision="float32"
            )
            print(f"Created index {self.collection_name}")
        except Exception as e:
            # Proceed if collection is likely already initialized
            print(f"Collection initialization error / might exist: {e}")
            pass
            
        vectors = []
        
        # Structure the payload
        for i, pattern in enumerate(dsa_patterns):
            # Rich text for semantic accuracy
            text_to_embed = f"{pattern['name']}: {pattern['description']} Use Cases: {pattern['use_cases']}"
            vector = self.model.encode(text_to_embed)
            
            vectors.append({
                "id": f"pattern_{i}",
                "vector": vector.tolist(),
                "meta": {
                    "name": pattern['name'],
                    "description": pattern['description'],
                    "use_cases": pattern['use_cases'],
                    "time_complexity": pattern['time_complexity'],
                    "space_complexity": pattern['space_complexity']
                }
            })
            
        # Strongly typed to Endee index.upsert() method
        if vectors:
            index = self.client.get_index(self.collection_name)
            index.upsert(vectors)
            
    def search_pattern(self, query: str, top_k: int = 2):
        """Embeds the query and performs a semantic similarity search in Endee."""
        query_vector = self.model.encode(query).tolist()
        
        index = self.client.get_index(self.collection_name)
        
        # Strictly adhering to Endee Index query() syntax
        results = index.query(
            vector=query_vector,
            top_k=top_k
        )
        return results
