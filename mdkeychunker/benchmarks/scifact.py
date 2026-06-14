"""
Industrial RAG Chunker Benchmark Suite.
Standardized evaluation on scientific and technical datasets.
"""
import time
import logging
import torch
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from beir.retrieval.evaluation import EvaluateRetrieval
from beir import util
from mdkeychunker import Config, Pipeline

logger = logging.getLogger(__name__)

def run_scifact_benchmark(model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu"):
    """
    Standardized benchmark on the SciFact dataset.
    Isolates chunking strategy as the only variable.
    """
    logger.info("Loading SciFact dataset...")
    corpus_ds = load_dataset("mteb/scifact", "corpus", split="corpus")
    queries_ds = load_dataset("mteb/scifact", "queries", split="queries")
    qrels_ds = load_dataset("mteb/scifact", split="test")

    corpus = {row['_id']: row['text'] for row in corpus_ds}
    queries = {row['_id']: row['text'] for row in queries_ds}
    qrels = {}
    for row in qrels_ds:
        if row['query-id'] not in qrels:
            qrels[row['query-id']] = {}
        qrels[row['query-id']][row['corpus-id']] = int(row['score'])

    encoder = SentenceTransformer(model_name, device=device)
    
    # Run MDKeyChunker Strategy
    logger.info("Executing MDKeyChunker Strategy...")
    config = Config(max_chunk_size=512, merge_by_keys=True)
    pipeline = Pipeline(config, enricher_mode="spacy")
    
    chunks_with_ids = []
    for doc_id, text in tqdm(corpus.items(), desc="Chunking"):
        chunks = pipeline.process_text(text)
        for c in chunks:
            chunks_with_ids.append((doc_id, c.text))
            
    # Flatten and Embed
    chunk_texts = [c[1] for c in chunks_with_ids]
    chunk_doc_ids = [c[0] for c in chunks_with_ids]
    
    logger.info(f"Embedding {len(chunk_texts)} chunks...")
    corpus_emb = encoder.encode(chunk_texts, show_progress_bar=True, batch_size=32, device=device, convert_to_tensor=True)
    
    logger.info("Embedding queries...")
    query_emb = encoder.encode(list(queries.values()), show_progress_bar=False, device=device, convert_to_tensor=True)
    
    cos_scores = util.cos_sim(query_emb, corpus_emb).cpu().numpy()
    
    results = {qid: {} for qid in queries.keys()}
    query_ids = list(queries.keys())
    for q_idx, qid in enumerate(query_ids):
        for c_idx, score in enumerate(cos_scores[q_idx]):
            doc_id = chunk_doc_ids[c_idx]
            if doc_id not in results[qid] or score > results[qid][doc_id]:
                results[qid][doc_id] = float(score)
    
    evaluator = EvaluateRetrieval(k_values=[1, 5, 10])
    ndcg, _, recall, _ = evaluator.evaluate(qrels, results, evaluator.k_values)
    
    print("\n" + "="*50)
    print("SCIFACT BENCHMARK RESULTS")
    print("="*50)
    print(f"nDCG@5:  {ndcg['NDCG@5']:.4f}")
    print(f"Recall@5: {recall['Recall@5']:.4f}")
    print(f"Avg Chunks/Doc: {len(chunks_with_ids)/len(corpus):.2f}")
    print("="*50)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    run_scifact_benchmark(device=dev)
