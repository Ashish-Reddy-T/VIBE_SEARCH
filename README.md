# Vibe Search™ - RAG-Based Place Recommender

A sophisticated place recommendation system built for the NYU Datathon Track 2 challenge. This implementation uses modern Retrieval Augmented Generation (RAG) techniques to find places in NYC that match specific "vibes" and intangible qualities.

## Key Features

- **Semantic Understanding**: Capture the true meaning behind queries like "where to find hot guys" or "cafes to cowork from"
- **True RAG Implementation**: Uses modern embedding models and LLMs for nuanced understanding
- **Vibe Detection**: Automatically identifies vibes and atmospheres from place data
- **Contextual Explanations**: Explains why each place matches your query
- **Fast Results**: Optimized to deliver results quickly despite complex processing
- **Beautiful UI**: Modern, responsive interface to display results

## Quick Start

```bash
# Install dependencies and run
python run_rag.py
```

Then open your browser to: http://localhost:8000

## Technologies Used

### Embedding Models
- **Sentence Transformers**: Modern semantic embedding models
  - Options: MiniLM, BGE, MPNet
  - Captures nuanced meaning in text

### Vector Database
- **FAISS**: High-performance similarity search 
  - Fast retrieval even with thousands of items
  - Efficient cosine similarity calculations

### LLM Integration
- **LangChain**: Framework for LLM applications
  - Structured prompts for context and explanations
  - Support for multiple LLM providers

### Optional LLMs
- **Local LLMs**: Via llama-cpp-python
  - Works offline with models like Llama, Mistral, etc.
- **OpenAI API**: For enhanced explanations
  - Set OPENAI_API_KEY in .env file to use

## Architecture

This system follows a true RAG (Retrieval Augmented Generation) architecture:

1. **Indexing Phase**:
   - Process place data, reviews, and media
   - Extract vibe attributes and enrich metadata
   - Generate high-quality embeddings
   - Build optimized vector index

2. **Retrieval Phase**:
   - Process user query with semantic understanding
   - Expand query to capture related concepts
   - Retrieve relevant places using vector similarity
   - Apply contextual filtering (neighborhoods, vibes)

3. **Generation Phase**:
   - Generate explanations for why places match the query
   - Provide context about each place's vibe and atmosphere
   - Create a cohesive, helpful response

## How Vibe Search Works

### 1. Ingestion & Embedding
The system processes place data from multiple sources:
- **Structured Data**: Name, location, tags, etc.
- **Unstructured Data**: Reviews, descriptions
- **Media**: Image URLs 

This data is processed to extract vibe attributes, which are combined with the original data and embedded using state-of-the-art models.

### 2. Query Understanding & Expansion
When you search for something like "cafes to cowork from", the system:
- **Understands the Concept**: Recognizes "coworking" implies wifi, quiet, outlets, etc.
- **Expands the Query**: Adds related terms to improve results
- **Extracts Constraints**: Identifies location filters, price ranges, etc.

### 3. Similarity Search & Filtering
The system searches for places that match the expanded query:
- **Vector Similarity**: Finds semantically similar places
- **Filtering**: Applies neighborhood and vibe filters
- **Ranking**: Orders results by relevance

### 4. Explanation Generation
For each result, the system generates an explanation:
- **Contextual Understanding**: Why this place matches your query
- **Highlights Features**: Emphasizes relevant attributes
- **Natural Language**: Presents information conversationally

## Advanced Usage

### Command Line Options

```bash
# Run with specific embedding model
python run_rag.py --embedding-model bge-small

# Build index before starting
python run_rag.py --build-index

# Use OpenAI for explanations (requires API key)
python run_rag.py --llm openai

# Run on specific port
python run_rag.py --port 8080
```

### Environment Variables

Create a `.env` file to configure:

```
EMBEDDING_MODEL=bge-small
LLM_PROVIDER=local
OPENAI_API_KEY=your_key_here
```

## Example Queries

- "cafes to cowork from"
- "matcha latte in the east village"
- "where can I spend a sunny day?"
- "what to do this weekend"
- "where to find hot guys?"
- "where to take someone on a second date?"
- "dance-y bars that have disco balls"

## Technical Details

### Embedding Models

This implementation supports multiple embedding models:

| Model | Dimensions | Speed | Quality |
|-------|------------|-------|---------|
| MiniLM | 384 | ⚡⚡⚡ | ⭐⭐ |
| BGE-Small | 384 | ⚡⚡⚡ | ⭐⭐⭐ |
| BGE-Base | 768 | ⚡⚡ | ⭐⭐⭐⭐ |
| BGE-Large | 1024 | ⚡ | ⭐⭐⭐⭐⭐ |
| MPNet | 768 | ⚡⚡ | ⭐⭐⭐⭐ |

### Vibe Categories

The system automatically categorizes places into vibes:

- date_night
- work_friendly
- outdoor_vibes
- group_hangout
- food_focus
- drinks_focus
- coffee_tea
- dancing_music
- quiet_relaxing
- upscale_fancy
- casual_lowkey
- unique_special
- trendy_cool
- budget_friendly

## Performance Optimizations

1. **Vectorization**: Fast similarity search with FAISS
2. **Cached Embeddings**: Reuse query embeddings for similar searches
3. **Cached Explanations**: Store explanations for common patterns
4. **Timeout Handling**: Enforce limits on search and explanation time
5. **Batched Processing**: Process data in optimal batches