import numpy as np
import json
import argparse
from pathlib import Path
import joblib
from audio_embeddings import (
    compute_embedding, 
    extract_audio_features,
    compute_similarity_scores
)
import requests
import time

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

PREVIEW_CACHE_DIR = Path('preview_cache')

def download_preview(url, track_id):
    if not url:
        return None
    
    possible_paths = [
        PREVIEW_CACHE_DIR / f"{track_id}.mp3",
        PREVIEW_CACHE_DIR / f"deezer_{track_id}.mp3",
    ]
    
    for cache_path in possible_paths:
        if cache_path.exists():
            if cache_path.stat().st_size > 0:
                return str(cache_path)
            else:
                print(f"cached file exists but is empty: {cache_path.name}")
    
    return None

def query_similar_tracks(
    query_audio_path,
    top_k=50,
    top_n=20,
    top_m=10,
    weights=(0.7, 0.2, 0.1)
):

    index_data = joblib.load('index_data.pkl')
    
    metadata = index_data['metadata']
    deezer_features = index_data['deezer_features']
    preview_embeddings = index_data['preview_embeddings']
    combined_features = index_data['combined_features']
    emb_dim = index_data['embedding_dim']
    has_faiss = index_data.get('has_faiss', False)
    
    deezer_dim = deezer_features.shape[1] if len(deezer_features) > 0 else 11
    expected_dim = deezer_dim + emb_dim
    
    print(f"Index dimensions: deezer={deezer_dim}, Embedding={emb_dim}, Total={expected_dim}")
    
    query_emb = compute_embedding(query_audio_path)
    if query_emb is None:
        raise ValueError("Failed to compute query embedding")
    
    # print(f"query embedding dimension: {query_emb.shape[0]}")
    
    query_emb_original = query_emb.copy()
    
    if query_emb.shape[0] != emb_dim:
        # print(f"WARNING: Query embedding dimension ({query_emb.shape[0]}) doesn't match index ({emb_dim})")
        if query_emb.shape[0] < emb_dim:
            query_emb = np.pad(query_emb, (0, emb_dim - query_emb.shape[0]), mode='constant')
            # print(f"padded query embedding to: {query_emb.shape[0]}")
        else:
            query_emb = query_emb[:emb_dim]
        # print(f"adjusted to dimension: {query_emb.shape[0]}")
    else:
        query_emb_original = query_emb
    
    query_features = extract_audio_features(query_audio_path)
    if query_features is None:
        # print("Warning: Could not extract query features")
        query_features = {}
    
    query_deezer = np.zeros(deezer_dim, dtype=np.float32)
    query_combined = np.concatenate([query_deezer, query_emb])
    query_combined = query_combined.reshape(1, -1)
    
    # print(f"Query vector shape: {query_combined.shape}")
    # print(f"Expected shape: (1, {expected_dim})")
    
    if query_combined.shape[1] != expected_dim:
        raise ValueError(
            f"Dimension mismatch! Query: {query_combined.shape[1]}, "
            f"Expected: {expected_dim}"
        )
    
    from sklearn.preprocessing import normalize
    query_combined = normalize(query_combined, axis=1)
    
    print(f"Retrieving top-{top_k} candidates...")
    
    if has_faiss and HAS_FAISS:
        index = faiss.read_index('music_index.faiss')
        # print(f"FAISS index dimension: {index.d}")
        
        if query_combined.shape[1] != index.d:
            raise ValueError(
                f"CRITICAL: Query dimension ({query_combined.shape[1]}) != "
                f"Index dimension ({index.d})"
            )
        
        dist, indices = index.search(query_combined.astype('float32'), top_k)
        candidates = indices[0].tolist()
    else:
        index = joblib.load('music_index.pkl')
        dist, indices = index.kneighbors(query_combined, n_neighbors=min(top_k, len(metadata)))
        candidates = indices[0].tolist()
    
    # print(f"Retrieved {len(candidates)} candidates")
    
    
    if not PREVIEW_CACHE_DIR.exists():
        print(f"WARNING: Cache directory doesn't exist: {PREVIEW_CACHE_DIR}")
    else:
        cached_files = list(PREVIEW_CACHE_DIR.glob("*.mp3"))
        # print(f"Found {len(cached_files)} files in cache directory")
        # if len(cached_files) > 0:
        #     print(f"Sample cached files: {[f.name for f in cached_files[:3]]}")
    
    reranked = []
    rerank_count = 0
    no_preview = 0
    not_in_cache = 0
    
    for idx in candidates:
        if rerank_count >= top_n:
            break
        
        track = metadata[idx]
        preview_url = track.get('preview_url')
        
        if not preview_url:
            no_preview += 1
            continue
        
        preview_path = download_preview(preview_url, track['id'])
        if not preview_path:
            not_in_cache += 1
            continue
        
        if preview_embeddings[idx] is None:
            preview_emb = compute_embedding(preview_path)
            if preview_emb is None:
                continue
        else:
            preview_emb = preview_embeddings[idx]
        
        q_emb = query_emb_original
        p_emb = preview_emb
        
        if q_emb.shape[0] != p_emb.shape[0]:
            min_dim = min(q_emb.shape[0], p_emb.shape[0])
            q_emb = q_emb[:min_dim]
            p_emb = p_emb[:min_dim]
        
        preview_sim = np.dot(q_emb, p_emb) / (
            np.linalg.norm(q_emb) * np.linalg.norm(p_emb) + 1e-8
        )
        
        candidate_features = extract_audio_features(preview_path)
        if candidate_features is None:
            candidate_features = {}
        
        candidate_deezer = track.get('features', {})
        scores = compute_similarity_scores(
            query_features,
            candidate_features,
            None,  
            candidate_deezer
        )
        
        q_tempo = query_features.get('tempo', 120)
        c_tempo = candidate_features.get('tempo', 120)
        tempo_diff = abs(q_tempo - c_tempo)
        tempo_sim = 1.0 / (1.0 + tempo_diff / 50.0)
        
        deezer_sim = 0.5
        
        w_preview, w_tempo, w_deezer = weights
        final_score = (
            w_preview * float(preview_sim) +
            w_tempo * tempo_sim +
            w_deezer * deezer_sim
        )
        print("\nCANDIDATE DEBUG:")
        print(repr(candidate_deezer))
        

        result = {
            'id': track['id'],
            'name': track.get('name') or track.get('title') or track.get('title_short'),
            'artists': (        track.get('artist') 
        if isinstance(track.get('artist'), str)
        else track.get('artist', {}).get('name')),
            'preview_url': preview_url,
            'score': float(final_score),
            'breakdown': {
                'preview_similarity': float(preview_sim),
                'tempo_similarity': float(tempo_sim),
                'deezer_similarity': float(deezer_sim),
                'timbre': float(scores['timbre']),
                'harmony': float(scores['harmony']),
                'rhythm': float(scores['rhythm'])
            },
            'features': {
                'tempo': float(c_tempo),
            }
        }
        
        reranked.append(result)
        rerank_count += 1
        
        if rerank_count % 5 == 0:
            print(f"  Reranked {rerank_count}/{top_n}...")
    
    # print(f"processed: {rerank_count}")
    # print(f"no preview URL: {no_preview}")
    # print(f"not in cache: {not_in_cache}")
    
    reranked.sort(key=lambda x: x['score'], reverse=True)
    
    results = reranked[:top_m]
    
    # print(f"\nfound {len(results)} similar tracks")
    
    return results

def format_results(results):
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['name']} - {result['artists']}")
        print(f"   Score: {result['score']:.3f}")
        print(f"   Preview: {result['preview_url']}")
        
        breakdown = result['breakdown']
        print(f"   Breakdown:")
        print(f"     - Preview similarity: {breakdown['preview_similarity']:.3f}")
        print(f"     - Timbre: {breakdown['timbre']:.3f}")
        print(f"     - Harmony: {breakdown['harmony']:.3f}")
        print(f"     - Rhythm: {breakdown['rhythm']:.3f}")
        print(f"     - Tempo similarity: {breakdown['tempo_similarity']:.3f}")
        
        features = result['features']
        print(f"   Features:")
        print(f"     - Tempo: {features['tempo']:.1f} BPM")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Query music similarity index')
    parser.add_argument('audio_file', help='Path to query audio file')
    parser.add_argument('--top-k', type=int, default=50, help='Coarse retrieval candidates')
    parser.add_argument('--top-n', type=int, default=20, help='Candidates to rerank')
    parser.add_argument('--top-m', type=int, default=10, help='Final results to return')
    parser.add_argument('--output', help='Output JSON file')
    parser.add_argument('--preview-weight', type=float, default=0.7)
    parser.add_argument('--tempo-weight', type=float, default=0.2)
    parser.add_argument('--deezer-weight', type=float, default=0.1)
    
    args = parser.parse_args()
    
    weights = (args.preview_weight, args.tempo_weight, args.deezer_weight)
    
    results = query_similar_tracks(
        args.audio_file,
        top_k=args.top_k,
        top_n=args.top_n,
        top_m=args.top_m,
        weights=weights
    )
    
    format_results(results)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nresults saved to {args.output}")