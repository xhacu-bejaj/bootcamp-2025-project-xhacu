import json
import os
import sys
from pathlib import Path
import multiprocessing
from tqdm import tqdm
from functools import partial
from typing import List, Dict, Any, Tuple, Iterable

# Add project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.chunk_store import ChunkStore
from app.services.chunking import ChunkingStrategyFactory

# --- Configuration ---
# Set the maximum length for each text chunk.
MAX_CHUNK_LENGTH = 512
# Set the number of chunks to batch together for a single database insertion.
DB_BATCH_SIZE = 4000
# Determine the number of parallel processes to use for chunking.
# Leaves one CPU core free for the main process and database work.
cpu_count = os.cpu_count()
NUM_PROCESSES = max(1, cpu_count - 1) if cpu_count else 1


def worker_chunk_article(article: Dict[str, Any]) -> List[Tuple[str, dict]]:
    """
    Worker function to process a single article into a list of chunks.
    This function is executed in a separate process and is purely CPU-bound.
    """
    # Each worker initializes its own chunking strategy instance.
    chunking_strategy = ChunkingStrategyFactory.get_strategy("semantic", max_length=MAX_CHUNK_LENGTH)
    
    source_id = article.get("id", "")
    title = article.get("title", "")
    abstract = article.get("abstract", "")

    if not source_id or (not title and not abstract):
        return []

    text_to_chunk = f"Title: {title}\n\nAbstract: {abstract}"
    
    try:
        chunks = chunking_strategy.chunk_text(text_to_chunk)
    except Exception as e:
        # If chunking fails for one article, we skip it and move on.
        print(f"Warning: Could not chunk article {source_id}. Error: {e}")
        return []

    if not chunks:
        return []

    base_metadata = {
        "source_id": source_id,
        "keywords": ", ".join(article.get("keywords", [])),
        "authors": ", ".join(article.get("authors", [])),
        "venue": ", ".join(article.get("venue", [])),
        "date": article.get("date", ""),
        "teams": ", ".join(article.get("Teams", []))
    }
    base_metadata = {k: v for k, v in base_metadata.items() if v}

    # Return a list of tuples, each containing the chunk text and its metadata.
    return [
        (chunk_text, {**base_metadata, "chunk_id": f"{source_id}-{i}"})
        for i, chunk_text in enumerate(chunks)
    ]

def main():
    """Main function to orchestrate the parallel data loading."""
    print("--- Starting High-Performance Dataset Loading Script ---")
    print(f"Using {NUM_PROCESSES} parallel processes for chunking.")

    # --- 1. Initialize ChunkStore in the Main Process ---
    print("Initializing ChunkStore...")
    chunk_store = ChunkStore()

    # --- 2. Clear Existing Data ---
    print(f"Clearing existing data from collection '{chunk_store._collection.name}'...")
    chunk_store.clear()
    print("Collection cleared successfully.")

    # --- 3. Load JSON Data ---
    json_path = project_root / "data.raw.json"
    print(f"Loading article data from '{json_path}'...")
    with open(json_path, "r", encoding="utf-8") as f:
        articles = json.load(f)
    print(f"Found {len(articles)} articles to process.")

    # --- 4. Parallel Processing and Batch Insertion ---
    chunk_texts_batch = []
    chunk_metadatas_batch = []
    total_chunk_count = 0

    # The `with` statement ensures the pool is properly closed.
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        # `imap_unordered` processes the articles in parallel and returns results
        # as they are completed, which is memory efficient.
        # The tqdm progress bar tracks the completion of articles.
        results_iterator = pool.imap_unordered(worker_chunk_article, articles)
        
        print("Starting parallel chunking and database insertion...")
        for article_chunks in tqdm(results_iterator, total=len(articles), desc="Processing Articles"):
            if not article_chunks:
                continue

            for text, meta in article_chunks:
                chunk_texts_batch.append(text)
                chunk_metadatas_batch.append(meta)

            # When the batch for the database is full, insert it.
            if len(chunk_texts_batch) >= DB_BATCH_SIZE:
                tqdm.write(f"\nInserting batch of {len(chunk_texts_batch)} chunks into database...")
                chunk_store.insert_chunks(texts=chunk_texts_batch, metadatas=chunk_metadatas_batch)
                total_chunk_count += len(chunk_texts_batch)
                # Clear the batch lists for the next set of chunks.
                chunk_texts_batch.clear()
                chunk_metadatas_batch.clear()
                tqdm.write("Batch inserted successfully.")
    
    # --- 5. Insert the Final Remaining Batch ---
    if chunk_texts_batch:
        print(f"Inserting final batch of {len(chunk_texts_batch)} chunks...")
        chunk_store.insert_chunks(texts=chunk_texts_batch, metadatas=chunk_metadatas_batch)
        total_chunk_count += len(chunk_texts_batch)
        print("Final batch inserted successfully.")

    # --- 6. Final Report ---
    print("\n--- Dataset Loading Complete ---")
    print(f"Successfully inserted a total of {total_chunk_count} chunks.")
    print(f"Processed {len(articles)} articles.")
    print("------------------------------------")

if __name__ == "__main__":
    # This is crucial for multiprocessing to work correctly, especially on Windows.
    multiprocessing.freeze_support()
    main()
