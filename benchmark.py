"""
HUMEM vs Dense RAG Benchmarking Harness
Validates O(1) Routing latency and catastrophic interference vs semantic blurring.
"""

import time
import torch
import faiss
import numpy as np
import pandas as pd
from humem import HUMEM

# Hyperparameters
NUM_TURNS = 1000
EMBEDDING_DIM = 2048 # Llama-3.2-1B dim

def generate_synthetic_dataset(turns):
    """Generates homogeneous data to force Semantic Blurring and Hash Collisions."""
    dataset = []
    for i in range(turns):
        # Creates highly overlapping structural sentences
        prompt = f"What is the access code for Protocol Omega {i}?"
        fact = f"The secure access code for Protocol Omega {i} is {np.random.randint(10000, 99999)}."
        dataset.append((prompt, fact))
    return dataset

def run_benchmark():
    print("Initializing RAG (FAISS IndexFlatIP) vs HUMEM (256 Neurons)...")
    
    # Init RAG
    rag_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    rag_vault = {}
    
    # Init HUMEM
    humem = HUMEM(M=256, k=16, eta=0.1)
    
    dataset = generate_synthetic_dataset(NUM_TURNS)
    results = []

    for turn, (prompt, fact) in enumerate(dataset):
        # ---------------------------------------------------------
        # 1. RAG PIPELINE
        # ---------------------------------------------------------
        # Write Phase
        t0 = time.perf_counter()
        x_numpy = humem.extract_latent_vector(fact).numpy().reshape(1, -1)
        faiss.normalize_L2(x_numpy)
        rag_index.add(x_numpy)
        rag_vault[turn] = fact
        rag_write_time = (time.perf_counter() - t0) * 1000

        # Read Phase
        query_vec = humem.extract_latent_vector(prompt).numpy().reshape(1, -1)
        faiss.normalize_L2(query_vec)
        
        t0 = time.perf_counter()
        distances, indices = rag_index.search(query_vec, 1)
        rag_retrieved = rag_vault.get(indices[0][0], "")
        rag_read_time = (time.perf_counter() - t0) * 1000
        
        rag_hit = 1 if rag_retrieved == fact else 0

        # ---------------------------------------------------------
        # 2. HUMEM PIPELINE
        # ---------------------------------------------------------
        # Write Phase
        t0 = time.perf_counter()
        payload, x_tensor, y_tensor, n_top = humem.route(fact)
        humem.consolidate(x_tensor, y_tensor, n_top, fact)
        humem_write_time = (time.perf_counter() - t0) * 1000
        
        # Read Phase
        t0 = time.perf_counter()
        humem_retrieved, _, _, _ = humem.route(prompt)
        humem_read_time = (time.perf_counter() - t0) * 1000
        
        humem_hit = 1 if humem_retrieved == fact else 0
        
        # ---------------------------------------------------------
        # LOGGING
        # ---------------------------------------------------------
        results.append({
            "Turn_ID": turn + 1,
            "RAG_Retrieval_Latency_ms": rag_read_time,
            "HUMEM_Retrieval_Latency_ms": humem_read_time,
            "RAG_Hit": rag_hit,
            "HUMEM_Hit": humem_hit
        })
        
        if (turn + 1) % 100 == 0:
            print(f"Processed {turn + 1}/{NUM_TURNS} turns...")

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("humem_benchmark_results.csv", index=False)
    print("\nBenchmark Complete! Data saved to humem_benchmark_results.csv")

if __name__ == "__main__":
    run_benchmark()
