import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from retrieval_pipeline import ValidationPipeline

# Load environment variables
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

# Initialize the optional metadata-filtering client only when configured.
client = None
if AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY:
    client = OpenAI(base_url=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_API_KEY)

# Initialize the vector retrieval pipeline
pipeline = ValidationPipeline()

def parse_index_file(index_path="index.txt"):
    """
    Parses index.txt and returns a dictionary mapping filenames 
    to their full metadata string blocks.
    """
    metadata_dict = {}
    if not os.path.exists(index_path):
        print(f"Warning: {index_path} not found.")
        return metadata_dict
        
    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Split the file into blocks using the separator defined in ingestion_pipeline.py
    blocks = content.split("-" * 50)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        
        # Extract the filename to use as the dictionary key
        lines = block.split("\n")
        filename = None
        for line in lines:
            if line.startswith("Filename: "):
                filename = line.replace("Filename: ", "").strip()
                break
        
        if filename:
            metadata_dict[filename] = block
            
    return metadata_dict

def run_agent(text_query: str = None, image_path: str = None, k: int = 5):
    """
    Executes text or image search. If a text query is provided, it uses 
    an LLM to filter the top-k results based on EXIF metadata.
    """
    if not text_query and not image_path:
        raise ValueError("Must provide either a text_query or an image_path.")
        
    # 1. Retrieve initial top-k results from ChromaDB.
    if text_query and image_path:
        text_results = pipeline.text_search(query_text=text_query, k=k)
        image_results = pipeline.reverse_image_search(image_path=image_path, k=k)

        # Combine modality rankings with reciprocal-rank fusion so either signal
        # can surface a match, while images ranked highly by both score best.
        scores = {}
        results_by_filename = {}
        for ranked_results in (text_results, image_results):
            if isinstance(ranked_results, str):
                continue
            for rank, result in enumerate(ranked_results or [], start=1):
                filename = result.get("filename")
                if not filename:
                    continue
                scores[filename] = scores.get(filename, 0.0) + 1.0 / (60 + rank)
                results_by_filename.setdefault(filename, result)
        results = [
            results_by_filename[filename]
            for filename in sorted(scores, key=scores.get, reverse=True)
        ]
        if not results:
            return text_results if isinstance(text_results, str) else image_results
    elif text_query:
        results = pipeline.text_search(query_text=text_query, k=k)
    else:
        results = pipeline.reverse_image_search(image_path=image_path, k=k)
        # If it's a pure image search, no text prompt exists to filter metadata against.
        return results
        
    # Handle edge cases where no results are found
    if isinstance(results, str) or not results:
        return results
        
    # 2. Extract metadata from index.txt for the specifically retrieved images
    metadata_map = parse_index_file("index.txt")
    context_blocks = []
    
    for i, res in enumerate(results):
        filename = res["filename"]
        meta_info = metadata_map.get(filename, f"Filename: {filename}\nNo EXIF metadata available.")
        context_blocks.append(f"Index [{i}]:\n{meta_info}")
        
    context_str = "\n\n".join(context_blocks)
    
    # 3. Formulate the LLM prompt for metadata filtering
    system_prompt = (
        "You are a helpful filtering assistant. The following images have been shortlisted using "
        "the CLIP algorithm. Your job is to specifically shortlist (if possible) the images based "
        "on any metadata requirements the user says (e.g., date, time, location, camera model, etc.).\n\n"
        "If no such metadata requirement is mentioned in the user's prompt, return the full set of "
        "image indexes back. Otherwise, return only the indexes that match the metadata requirement.\n\n"
        "You must respond strictly in JSON format with a single key 'indexes' mapped to a list of integers. "
        "Example: {\"indexes\": [0, 2, 4]}"
    )
    
    user_prompt = f"User Prompt: {text_query}\n\nRetrieved Images Metadata:\n{context_str}"
    
    # 4. Execute LLM Filtering
    if client is None or not DEPLOYMENT_NAME:
        print("[Agent] Azure OpenAI is not configured; returning CLIP retrieval results.")
        return results

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        # 5. Parse structured output and filter results
        llm_output = response.choices[0].message.content
        parsed_output = json.loads(llm_output)
        selected_indexes = parsed_output.get("indexes", list(range(len(results))))
        
        # Build the final filtered list, ensuring indexes are within bounds
        final_results = [
            results[i]
            for i in selected_indexes
            if isinstance(i, int) and 0 <= i < len(results)
        ]
        
        print(f"\n[Agent] Original CLIP results: {len(results)}. LLM filtered down to: {len(final_results)}.")
        return final_results
        
    except Exception as e:
        print(f"[Agent] Error during LLM metadata filtering: {e}")
        # Fail gracefully by returning the unfiltered CLIP results
        return results

# ==========================================
# Example Usage / Testing Block
# ==========================================
if __name__ == "__main__":
    # Example Usage
    text_query = "photo of an equation in a yellow"
    results = run_agent(text_query=text_query, k=5)
    
    print("\nFiltered Results:")
    for res in results:
        print(f"- {res['filename']} (Score: {res['distance_score']})")