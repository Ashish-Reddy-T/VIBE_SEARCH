"""
RAG indexing script for Vibe Search
Uses sentence-transformers for semantic understanding of place data
"""
import os
import pandas as pd
import numpy as np
import pickle
import faiss
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Constants
EMBED_MODELS = {
    'minilm': 'all-MiniLM-L6-v2',    # Fast, good for testing (384 dimensions)
    'bge-small': 'BAAI/bge-small-en-v1.5',  # Fast, good quality (384 dimensions)
    'bge-base': 'BAAI/bge-base-en-v1.5',    # Good balance (768 dimensions)
    'bge-large': 'BAAI/bge-large-en-v1.5',  # High quality but slower (1024 dimensions)
    'mpnet': 'all-mpnet-base-v2',     # High quality (768 dimensions)
}

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLACES_CSV = os.path.join(BASE_DIR, 'places.csv')
REVIEWS_CSV = os.path.join(BASE_DIR, 'reviews.csv')
MEDIA_CSV = os.path.join(BASE_DIR, 'media.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'ingestion')

# Vibe categories for metadata enrichment
VIBE_CATEGORIES = {
    "date_night": [
        "romantic", "intimate", "cozy", "candlelit", "date", "charming"
    ],
    "work_friendly": [
        "wifi", "laptop", "outlets", "quiet", "work", "study", "spacious"
    ],
    "outdoor_vibes": [
        "outdoor", "patio", "terrace", "garden", "rooftop", "fresh air",
        "alfresco", "sunshine", "sunny"
    ],
    "group_hangout": [
        "group", "friends", "party", "social", "gathering", "fun", "lively"
    ],
    "food_focus": [
        "food", "restaurant", "delicious", "menu", "chef", "cuisine", "eat",
        "dining", "dinner", "lunch", "brunch"
    ],
    "drinks_focus": [
        "bar", "drink", "cocktail", "beer", "wine", "alcohol", "happy hour"
    ],
    "coffee_tea": [
        "coffee", "cafe", "espresso", "latte", "cappuccino", "tea", "matcha"
    ],
    "dancing_music": [
        "dance", "club", "dj", "music", "live", "performance", "party"
    ],
    "quiet_relaxing": [
        "quiet", "peaceful", "calm", "relaxing", "tranquil", "serene"
    ],
    "upscale_fancy": [
        "upscale", "fancy", "elegant", "luxury", "high-end", "fine dining"
    ],
    "casual_lowkey": [
        "casual", "relaxed", "laid-back", "informal", "simple"
    ],
    "unique_special": [
        "unique", "special", "quirky", "interesting", "eclectic", "hidden gem"
    ],
    "trendy_cool": [
        "trendy", "hip", "cool", "stylish", "instagram", "fashionable"
    ],
    "budget_friendly": [
        "affordable", "cheap", "budget", "inexpensive", "reasonable"
    ]
}

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Build vector index for Vibe Search")
    parser.add_argument(
        "--model", 
        choices=list(EMBED_MODELS.keys()), 
        default="minilm",
        help="Embedding model to use"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="Batch size for embedding generation"
    )
    return parser.parse_args()

def clean_tags(tags_str):
    """Clean tags string from the CSV format"""
    if pd.isna(tags_str):
        return []
    # Convert "{tag1,tag2}" format to list
    tags = tags_str.replace("{", "").replace("}", "").split(",")
    return [t.strip() for t in tags if t.strip()]

def extract_vibe_tags(text):
    """Extract vibe tags based on keyword matching"""
    text = text.lower()
    detected_vibes = []
    
    for vibe_name, keywords in VIBE_CATEGORIES.items():
        for keyword in keywords:
            if keyword in text:
                detected_vibes.append(vibe_name)
                break
                
    return detected_vibes

def create_document(place, reviews_df):
    """Create a rich text representation of a place"""
    # Basic place info
    name = str(place['name']) if pd.notna(place['name']) else ''
    neighborhood = str(place['neighborhood']) if pd.notna(place['neighborhood']) else ''
    desc = str(place['short_description']) if pd.notna(place['short_description']) else ''
    
    # Clean and process tags
    tags = clean_tags(place.get('tags', ''))
    tags_str = ' '.join(tags)
    
    # Get reviews for this place
    place_reviews = reviews_df[reviews_df['place_id'] == place['place_id']]['review_text'].tolist()
    reviews_text = ' '.join([str(r) for r in place_reviews[:5] if pd.notna(r)])
    
    # Create a rich document for embedding
    doc = f"Place: {name}. {desc}. Located in {neighborhood}. Type: {tags_str}. Reviews: {reviews_text[:500]}"
    return doc

def main():
    args = parse_args()
    model_name = EMBED_MODELS[args.model]
    batch_size = args.batch_size
    
    # Output files based on model name
    index_file = os.path.join(OUTPUT_DIR, f"index_{args.model}.faiss")
    metadata_file = os.path.join(OUTPUT_DIR, f"metadata_{args.model}.pkl")
    
    print(f"Loading data from {BASE_DIR}...")
    places_df = pd.read_csv(PLACES_CSV)
    reviews_df = pd.read_csv(REVIEWS_CSV)
    media_df = pd.read_csv(MEDIA_CSV)
    
    print(f"Loaded {len(places_df)} places, {len(reviews_df)} reviews, {len(media_df)} media items")
    
    # Prepare documents and extract features
    print("Creating documents and extracting features...")
    documents = []
    metadata = []
    
    for _, row in tqdm(places_df.iterrows(), total=len(places_df), desc="Preparing documents"):
        # Create document
        doc = create_document(row, reviews_df)
        documents.append(doc)
        
        # Get media URLs (just first one for simplicity)
        media_urls = media_df[media_df['place_id'] == row['place_id']]['media_url'].tolist()
        image_url = media_urls[0] if media_urls else ''
        
        # Extract vibe tags
        combined_text = f"{row['name']} {row['short_description']} {row.get('tags', '')} "
        combined_text += ' '.join(reviews_df[reviews_df['place_id'] == row['place_id']]['review_text'].head(5).tolist())
        vibe_tags = extract_vibe_tags(combined_text)
        
        # Prepare metadata
        metadata.append({
            'place_id': str(row['place_id']),
            'name': str(row['name']) if pd.notna(row['name']) else '',
            'neighborhood': str(row['neighborhood']) if pd.notna(row['neighborhood']) else '',
            'latitude': float(row['latitude']),
            'longitude': float(row['longitude']),
            'emoji': str(row['emoji']) if pd.notna(row['emoji']) else '📍',
            'short_description': str(row['short_description']) if pd.notna(row['short_description']) else '',
            'tags': clean_tags(row.get('tags', '')),
            'image_url': image_url,
            'vibe_tags': vibe_tags
        })
    
    # Initialize the embedding model
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Generate embeddings in batches
    print("Generating embeddings...")
    embeddings = []
    for i in tqdm(range(0, len(documents), batch_size), desc="Embedding"):
        batch = documents[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.append(batch_embeddings)
    
    # Combine batches
    embeddings = np.vstack(embeddings).astype('float32')
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create and save FAISS index
    print("Creating FAISS index...")
    d = embeddings.shape[1]  # embedding dimension
    index = faiss.IndexFlatIP(d)  # Inner product == cosine on normalized vectors
    index.add(embeddings)
    
    # Save index and metadata
    print(f"Saving index to {index_file}")
    faiss.write_index(index, index_file)
    
    print(f"Saving metadata to {metadata_file}")
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    print("Done!")

if __name__ == "__main__":
    main()