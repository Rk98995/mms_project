import os
import torch
import chromadb
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Configuration
CHROMA_DB_DIR = "./chroma_db"
LOCAL_MODEL_DIR = "./local_clip_model"

class ValidationPipeline:
    def __init__(self):
        print("Initializing Validation Pipeline...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load CLIP Model (Prefer local cache)
        if os.path.exists(LOCAL_MODEL_DIR):
            print(f"Loading CLIP model offline from {LOCAL_MODEL_DIR}...")
            self.model = CLIPModel.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True)
        else:
            print("Local model not found. Please run the ingestion pipeline first to cache the model.")
            model_id = "openai/clip-vit-base-patch32"
            self.model = CLIPModel.from_pretrained(model_id).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_id)

        # Initialize ChromaDB Client
        print(f"Connecting to ChromaDB at {CHROMA_DB_DIR}...")
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.collection = self.chroma_client.get_collection(name="image_embeddings")
        print("Pipeline ready.\n")

    def text_search(self, query_text: str, k: int = 5):
        """Searches for images matching a text query."""
        print(f"--- Text Search: '{query_text}' (Top {k}) ---")
        
        # Process and encode text
        inputs = self.processor(text=[query_text], padding=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            
            # Safe extraction
            if hasattr(outputs, 'pooler_output'):
                text_features = outputs.pooler_output
            elif hasattr(outputs, 'text_embeds'):
                text_features = outputs.text_embeds
            elif isinstance(outputs, torch.Tensor):
                text_features = outputs
            else:
                text_features = outputs[0]

        # Normalize embedding
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        embedding = text_features.cpu().numpy().tolist()[0]

        # Query ChromaDB
        # Note: Since embeddings are normalized, Chroma's default L2 distance 
        # perfectly correlates with Cosine Similarity ranking.
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=k
        )
        
        return self._format_results(results)

    def reverse_image_search(self, image_path: str, k: int = 5):
        """Searches for visually similar images using an input image."""
        print(f"--- Reverse Image Search: '{image_path}' (Top {k}) ---")
        
        if not os.path.exists(image_path):
            return f"Error: Image path '{image_path}' does not exist."

        # Load and process image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return f"Error loading image: {e}"

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            
            # Safe extraction
            if hasattr(outputs, 'pooler_output'):
                image_features = outputs.pooler_output
            elif hasattr(outputs, 'image_embeds'):
                image_features = outputs.image_embeds
            elif isinstance(outputs, torch.Tensor):
                image_features = outputs
            else:
                image_features = outputs[0]

        # Normalize embedding
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        embedding = image_features.cpu().numpy().tolist()[0]

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=k
        )
        
        return self._format_results(results)

    def _format_results(self, results):
        """Helper method to cleanly format ChromaDB query outputs."""
        formatted_output = []
        
        if not results['ids'][0]:
            return "No results found."

        for i in range(len(results['ids'][0])):
            doc_id = results['ids'][0][i]
            distance = results['distances'][0][i]
            metadata = results['metadatas'][0][i]
            
            formatted_output.append({
                "rank": i + 1,
                "filename": doc_id,
                "distance_score": round(distance, 4),
                "filepath": metadata.get("filepath", "Unknown")
            })
            
        return formatted_output
