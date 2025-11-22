"""
Query the music similarity index with a local audio file.
Usage: python query_script.py <audio_file> [--top-k 50] [--top-n 20] [--output results.json]
"""
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

# Try FAISS
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

PREVIEW_CACHE_DIR = Path('preview_cache')

def download_preview(url, track_id):
    """Download preview clip if not cached."""
    if not url:
        return None
    
    # Try multiple naming patterns for cached files
    possible_paths = [
        PREVIEW_CACHE_DIR / f"{track_id}.mp3",
        PREVIEW_CACHE_DIR / f"deezer_{track_id}.mp3",
    ]
    
    for cache_path in possible_paths:
        if cache_path.exists():
            # Verify file is not empty
            if cache_path.stat().st_size > 0:
                return str(cache_path)
            else:
                print(f"Cached file exists but is empty: {cache_path.name}")
    
    # Don't try to download - Deezer URLs expire
    # If not in cache, skip this track
    return None

def query_similar_tracks(
    query_audio_path,
    top_k=50,
    top_n=20,
    top_m=10,
    weights=(0.7, 0.2, 0.1)
):
    """
    Find similar tracks using hybrid retrieval + reranking.
    
    Args:
        query_audio_path: Path to query audio file
        top_k: Number of candidates from coarse retrieval
        top_n: Number of candidates to rerank (download previews)
        top_m: Final number of results to return
        weights: (preview_weight, tempo_weight, spotify_weight)
    
    Returns:
        List of results with similarity breakdowns
    """
    print(f"Loading query: {query_audio_path}")
    
    # Load index
    print("Loading index...")
    index_data = joblib.load('index_data.pkl')
    
    metadata = index_data['metadata']
    spotify_features = index_data['spotify_features']
    preview_embeddings = index_data['preview_embeddings']
    combined_features = index_data['combined_features']
    emb_dim = index_data['embedding_dim']
    has_faiss = index_data.get('has_faiss', False)
    
    # Determine Spotify feature dimension from actual data
    spotify_dim = spotify_features.shape[1] if len(spotify_features) > 0 else 11
    expected_dim = spotify_dim + emb_dim
    
    print(f"Index dimensions: Spotify={spotify_dim}, Embedding={emb_dim}, Total={expected_dim}")
    
    # Compute query embedding
    print("Computing query embedding...")
    query_emb = compute_embedding(query_audio_path)
    if query_emb is None:
        raise ValueError("Failed to compute query embedding")
    
    print(f"Query embedding dimension: {query_emb.shape[0]}")
    
    # Keep original embedding for preview similarity
    query_emb_original = query_emb.copy()
    
    # Check if embedding dimension matches
    if query_emb.shape[0] != emb_dim:
        print(f"WARNING: Query embedding dimension ({query_emb.shape[0]}) doesn't match index ({emb_dim})")
        print("Adjusting query embedding for index search...")
        if query_emb.shape[0] < emb_dim:
            # Pad with zeros
            query_emb = np.pad(query_emb, (0, emb_dim - query_emb.shape[0]), mode='constant')
        else:
            # Truncate
            query_emb = query_emb[:emb_dim]
        print(f"Adjusted to dimension: {query_emb.shape[0]}")
    else:
        query_emb_original = query_emb  # They're the same
    
    query_features = extract_audio_features(query_audio_path)
    if query_features is None:
        print("Warning: Could not extract query features")
        query_features = {}
    
    # Build query vector (match index structure exactly)
    query_spotify = np.zeros(spotify_dim, dtype=np.float32)
    query_combined = np.concatenate([query_spotify, query_emb])
    query_combined = query_combined.reshape(1, -1)
    
    print(f"Query vector shape: {query_combined.shape}")
    print(f"Expected shape: (1, {expected_dim})")
    
    # Verify dimensions match
    if query_combined.shape[1] != expected_dim:
        raise ValueError(
            f"Dimension mismatch! Query: {query_combined.shape[1]}, "
            f"Expected: {expected_dim}"
        )
    
    # Normalize
    from sklearn.preprocessing import normalize
    query_combined = normalize(query_combined, axis=1)
    
    # Coarse retrieval
    print(f"Retrieving top-{top_k} candidates...")
    
    if has_faiss and HAS_FAISS:
        index = faiss.read_index('music_index.faiss')
        print(f"FAISS index dimension: {index.d}")
        
        # Final dimension check
        if query_combined.shape[1] != index.d:
            raise ValueError(
                f"CRITICAL: Query dimension ({query_combined.shape[1]}) != "
                f"Index dimension ({index.d})"
            )
        
        distances, indices = index.search(query_combined.astype('float32'), top_k)
        candidates = indices[0].tolist()
    else:
        index = joblib.load('music_index.pkl')
        distances, indices = index.kneighbors(query_combined, n_neighbors=min(top_k, len(metadata)))
        candidates = indices[0].tolist()
    
    print(f"Retrieved {len(candidates)} candidates")
    
    # Reranking phase: compute detailed similarities for top-N
    print(f"\nReranking top-{top_n} candidates with preview embeddings...")
    
    # Check cache directory
    if not PREVIEW_CACHE_DIR.exists():
        print(f"WARNING: Cache directory doesn't exist: {PREVIEW_CACHE_DIR}")
    else:
        cached_files = list(PREVIEW_CACHE_DIR.glob("*.mp3"))
        print(f"Found {len(cached_files)} files in cache directory")
        if len(cached_files) > 0:
            print(f"Sample cached files: {[f.name for f in cached_files[:3]]}")
    
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
        
        # Download preview if needed
        preview_path = download_preview(preview_url, track['id'])
        if not preview_path:
            not_in_cache += 1
            continue
        
        # Compute preview embedding if not cached
        if preview_embeddings[idx] is None:
            preview_emb = compute_embedding(preview_path)
            if preview_emb is None:
                continue
        else:
            preview_emb = preview_embeddings[idx]
        
        # Ensure embeddings have matching dimensions for similarity
        # Use the original query embedding (not padded)
        q_emb = query_emb_original
        p_emb = preview_emb
        
        # Adjust dimensions if needed
        if q_emb.shape[0] != p_emb.shape[0]:
            min_dim = min(q_emb.shape[0], p_emb.shape[0])
            q_emb = q_emb[:min_dim]
            p_emb = p_emb[:min_dim]
        
        # Compute preview similarity
        preview_sim = np.dot(q_emb, p_emb) / (
            np.linalg.norm(q_emb) * np.linalg.norm(p_emb) + 1e-8
        )
        
        # Extract candidate features
        candidate_features = extract_audio_features(preview_path)
        if candidate_features is None:
            candidate_features = {}
        
        # Compute detailed similarity scores
        candidate_spotify = track.get('features', {})
        scores = compute_similarity_scores(
            query_features,
            candidate_features,
            None,  # Query Spotify features not available
            candidate_spotify
        )
        
        # Tempo similarity
        q_tempo = query_features.get('tempo', 120)
        c_tempo = candidate_features.get('tempo', 120)
        tempo_diff = abs(q_tempo - c_tempo)
        tempo_sim = 1.0 / (1.0 + tempo_diff / 50.0)
        
        # Spotify feature similarity (simple)
        spotify_sim = 0.5  # Default when query Spotify features unavailable
        
        # Weighted combination
        w_preview, w_tempo, w_spotify = weights
        final_score = (
            w_preview * float(preview_sim) +
            w_tempo * tempo_sim +
            w_spotify * spotify_sim
        )
        
        
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
                'spotify_similarity': float(spotify_sim),
                'timbre': float(scores['timbre']),
                'harmony': float(scores['harmony']),
                'rhythm': float(scores['rhythm'])
            },
            'features': {
                'tempo': float(c_tempo),
                'danceability': candidate_spotify.get('danceability', 0),
                'energy': candidate_spotify.get('energy', 0),
                'valence': candidate_spotify.get('valence', 0)
            }
        }
        
        reranked.append(result)
        rerank_count += 1
        
        if rerank_count % 5 == 0:
            print(f"  Reranked {rerank_count}/{top_n}...")
    
    print(f"\nReranking statistics:")
    print(f"  - Processed: {rerank_count}")
    print(f"  - No preview URL: {no_preview}")
    print(f"  - Not in cache: {not_in_cache}")
    
    # Sort by final score
    reranked.sort(key=lambda x: x['score'], reverse=True)
    
    # Return top-M
    results = reranked[:top_m]
    
    print(f"\n✓ Found {len(results)} similar tracks")
    
    return results

def format_results(results):
    """Pretty print results."""
    print("\n" + "="*80)
    print("SIMILAR TRACKS")
    print("="*80)
    
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
        print(f"     - Danceability: {features['danceability']:.2f}")
        print(f"     - Energy: {features['energy']:.2f}")
        print(f"     - Valence: {features['valence']:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Query music similarity index')
    parser.add_argument('audio_file', help='Path to query audio file')
    parser.add_argument('--top-k', type=int, default=50, help='Coarse retrieval candidates')
    parser.add_argument('--top-n', type=int, default=20, help='Candidates to rerank')
    parser.add_argument('--top-m', type=int, default=10, help='Final results to return')
    parser.add_argument('--output', help='Output JSON file')
    parser.add_argument('--preview-weight', type=float, default=0.7)
    parser.add_argument('--tempo-weight', type=float, default=0.2)
    parser.add_argument('--spotify-weight', type=float, default=0.1)
    
    args = parser.parse_args()
    
    weights = (args.preview_weight, args.tempo_weight, args.spotify_weight)
    
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
        print(f"\n✓ Results saved to {args.output}")