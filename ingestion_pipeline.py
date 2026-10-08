import os
import torch
import chromadb
from PIL import Image, ExifTags
from transformers import CLIPProcessor, CLIPModel

# Configuration
IMAGE_DIR = "./gallery"  # <--- Replace with your placeholder path
CHROMA_DB_DIR = "./chroma_db"
INDEX_FILE = "index.txt"
LOCAL_MODEL_DIR = "./local_clip_model"   # <--- Local cache directory

def extract_exif(image):
    """Extracts EXIF metadata from an image and converts it to a serializable string dictionary."""
    exif_data = {}
    try:
        exif_info = image._getexif()
        if exif_info:
            for tag_id, value in exif_info.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                exif_data[str(tag)] = str(value)
    except AttributeError:
        pass
    except Exception as e:
        print(f"Could not extract EXIF: {e}")
    return exif_data

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai/clip-vit-base-patch32"

    # 1. Local Caching Logic for CLIP Model
    if os.path.exists(LOCAL_MODEL_DIR):
        print(f"Loading CLIP model offline from {LOCAL_MODEL_DIR}...")
        model = CLIPModel.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True).to(device)
        processor = CLIPProcessor.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True)
    else:
        print(f"Downloading CLIP model from Hugging Face... (This will only happen once)")
        model = CLIPModel.from_pretrained(model_id)
        processor = CLIPProcessor.from_pretrained(model_id)
        
        print(f"Saving model locally to {LOCAL_MODEL_DIR}...")
        model.save_pretrained(LOCAL_MODEL_DIR)
        processor.save_pretrained(LOCAL_MODEL_DIR)
        
        model = model.to(device)

    # 2. Initialize ChromaDB Persistent Client
    print(f"Initializing ChromaDB at {CHROMA_DB_DIR}...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = chroma_client.get_or_create_collection(name="image_embeddings")

    index_records = []
    
    if not os.path.isdir(IMAGE_DIR):
        print(f"Error: The directory {IMAGE_DIR} does not exist.")
        return

    # 3. Process Images
    valid_extensions = (".jpg", ".jpeg", ".png")
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(valid_extensions)]

    print(f"Found {len(image_files)} images. Starting ingestion...")

    for img_name in image_files:
        img_path = os.path.join(IMAGE_DIR, img_name)
        try:
            # Load Image
            image = Image.open(img_path).convert("RGB")

            # Extract EXIF Metadata
            exif_metadata = extract_exif(image)

            # Generate CLIP Embedding
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
                
                # Safely extract the raw tensor depending on what the library returns
                if hasattr(outputs, 'pooler_output'):
                    image_features = outputs.pooler_output
                elif hasattr(outputs, 'image_embeds'):
                    image_features = outputs.image_embeds
                elif isinstance(outputs, torch.Tensor):
                    image_features = outputs
                else:
                    image_features = outputs[0]  # Fallback if returned as a tuple

            # Normalize the embedding (standard practice for cosine similarity retrieval)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            embedding = image_features.cpu().numpy().tolist()[0]

            # Normalize embedding
            image_features /= image_features.norm(dim=-1, keepdim=True)
            embedding = image_features.cpu().numpy().tolist()[0]

            # Prepare metadata
            metadata_to_store = {"filename": img_name, "filepath": img_path}
            metadata_to_store.update(exif_metadata)

            # Store in ChromaDB
            collection.upsert(
                ids=[img_name],
                embeddings=[embedding],
                metadatas=[metadata_to_store]
            )

            # Record for index.txt
            index_records.append({
                "filename": img_name,
                "filepath": img_path,
                "exif_metadata": exif_metadata
            })

            print(f"  [✓] Indexed: {img_name}")

        except Exception as e:
            print(f"  [x] Error processing {img_name}: {e}")

    # 4. Write index.txt
    print(f"Writing metadata to {INDEX_FILE}...")
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        for record in index_records:
            f.write(f"Filename: {record['filename']}\n")
            f.write(f"Filepath: {record['filepath']}\n")
            f.write("EXIF Metadata:\n")
            if record['exif_metadata']:
                for k, v in record['exif_metadata'].items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write("  None\n")
            f.write("-" * 50 + "\n")

    print("Pipeline execution complete.")

if __name__ == "__main__":
    main()