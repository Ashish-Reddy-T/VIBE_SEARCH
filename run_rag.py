#!/usr/bin/env python
"""
Run script for the Vibe Search RAG application
Handles setup, dependency checking, and server startup
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

# Default configuration
DEFAULT_PORT = 8000
DEFAULT_HOST = "0.0.0.0"
DEFAULT_EMBED_MODEL = "minilm"  # Options: minilm, bge-small, bge-base, bge-large, mpnet

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run the Vibe Search RAG application")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to run the server on")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Host to bind to")
    parser.add_argument("--embedding-model", type=str, choices=["minilm", "bge-small", "bge-base", "bge-large", "mpnet"], 
                      default=DEFAULT_EMBED_MODEL, help="Embedding model to use")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload on file changes")
    parser.add_argument("--build-index", action="store_true", help="Build the search index before starting")
    return parser.parse_args()

def check_dependencies():
    """Check if required dependencies are installed"""
    required = [
        "fastapi", "uvicorn", "numpy", "pandas", "faiss-cpu",
        "sentence_transformers", "python-dotenv"
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        install = input("Install missing dependencies? (y/n): ")
        if install.lower() == "y":
            try:
                subprocess.run([sys.executable, "-m", "pip", "install"] + missing, check=True)
                print("Dependencies installed.")
            except Exception as e:
                print(f"Failed to install dependencies: {e}")
                sys.exit(1)
        else:
            print("Cannot continue without required dependencies.")
            sys.exit(1)

def check_index(model_name):
    """Check if the index for the specified model exists"""
    index_path = os.path.join("ingestion", f"index_{model_name}.faiss")
    metadata_path = os.path.join("ingestion", f"metadata_{model_name}.pkl")
    
    if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
        print(f"Index files for model '{model_name}' not found.")
        build = input("Build the index now? (y/n): ")
        if build.lower() == "y":
            build_index(model_name)
        else:
            print("Warning: Running without index files. Search will not work properly.")

def build_index(model_name):
    """Build the index for the specified model"""
    print(f"Building index for model '{model_name}'...")
    try:
        cmd = [sys.executable, "ingestion/rag_index.py", "--model", model_name]
        subprocess.run(cmd, check=True)
        print("Index built successfully!")
    except Exception as e:
        print(f"Failed to build index: {e}")
        print("Warning: Running without index files. Search will not work properly.")

def setup_environment(args):
    """Set up environment variables for the application"""
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    env_vars = {}
    
    if env_file.exists():
        # Read existing .env file
        with open(env_file, "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    env_vars[key] = value
    
    # Update environment variables
    env_vars["EMBEDDING_MODEL"] = args.embedding_model
    
    # Write updated .env file
    with open(env_file, "w") as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
    
    print(f"Environment configured with embedding model: {args.embedding_model}")

def main():
    """Main function to run the application"""
    args = parse_args()
    
    print("Starting Vibe Search RAG application...")
    
    # Check dependencies
    check_dependencies()
    
    # Set up environment
    setup_environment(args)
    
    # Check or build index if requested
    if args.build_index:
        build_index(args.embedding_model)
    else:
        check_index(args.embedding_model)
    
    # Run the application
    try:
        cmd = [
            sys.executable, "-m", "uvicorn", "rag_app:app",
            "--host", args.host,
            "--port", str(args.port)
        ]
        
        if not args.no_reload:
            cmd.append("--reload")
        
        print(f"\nStarting web server on {args.host}:{args.port}...")
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"\nError starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()