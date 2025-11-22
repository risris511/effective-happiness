import numpy as np
import json
from pathlib import Path
import joblib
from audio_embeddings import compute_embedding
import requests
import time

try:
    import faiss
    HAS_FAISS = True
    # print("Using FAISS for indexing")
except ImportError:
    from sklearn.neighbors import NearestNeighbors
    HAS_FAISS = False
    # print("Using sklearn for indexing")

PREVIEW_CACHE_DIR = Path('preview_cache')
PREVIEW_CACHE_DIR.mkdir(exist_ok=True)

def download_preview(url, track_id):
    if not url:
        return None
    
    cache_path = PREVIEW_CACHE_DIR / f"{track_id}.mp3"
    if cache_path.exists():
        return str(cache_path)
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        with open(cache_path, 'wb') as f:
            f.write(response.content)
        
        time.sleep(0.1)
        return str(cache_path)
    except Exception as e:
        print(f"failed to download preview for {track_id}: {e}")
        return None

def build_index(deezer_file='deezer_db.npz', max_previews=500):
    data = np.load(deezer_file, allow_pickle=True)
    
    deezer_features = data['features']
    metadata = json.loads(str(data['metadata']))
    feature_keys = data['feature_keys']
    
    # print(f"Loaded {len(metadata)} tracks")

    # print(f"\nComputing preview embeddings (max {max_previews})")
    preview_embeddings = []
    valid_indices = []
    
    preview_count = 0
    for i, track in enumerate(metadata):
        if preview_count >= max_previews:
            preview_embeddings.append(None)
            continue
        
        preview_url = track.get('preview_url')
        if not preview_url:
            preview_embeddings.append(None)
            continue
        
        preview_path = download_preview(preview_url, track['id'])
        if preview_path:
            emb = compute_embedding(preview_path)
            if emb is not None:
                preview_embeddings.append(emb)
                valid_indices.append(i)
                preview_count += 1
                
                # if preview_count % 50 == 0:
                #     # print(f"  processed {preview_count} previews")
                # continue
        
        preview_embeddings.append(None)
    
    # print(f"computed {preview_count} preview embeddings")
    
    valid_embs = [e for e in preview_embeddings if e is not None]
    if valid_embs:
        emb_dim = valid_embs[0].shape[0]
        # print(f"Preview embedding dimension: {emb_dim}")
    else:
        emb_dim = 512
        # print("No valid embeddings, using placeholder dimension")
    
    # Build combined feature vectors
    # print("\nBuilding combined feature vectors...")
    combined_features = []
    
    for i in range(len(metadata)):
        deezer_vec = deezer_features[i]
        preview_vec = preview_embeddings[i]
        
        if preview_vec is not None:
            combined = np.concatenate([deezer_vec, preview_vec])
        else:
            zero_pad = np.zeros(emb_dim)
            combined = np.concatenate([deezer_vec, zero_pad])
        
        combined_features.append(combined)
    
    combined_features = np.array(combined_features, dtype=np.float32)
    # print(f"Combined feature shape: {combined_features.shape}")

    from sklearn.preprocessing import normalize
    combined_features = normalize(combined_features, axis=1)
    
    
    if HAS_FAISS:
        d = combined_features.shape[1]
        index = faiss.IndexFlatIP(d)  # inner product (cosine after normalization)
        index.add(combined_features)

        faiss.write_index(index, 'music_index.faiss')
        # print("saved FAISS index to music_index.faiss")
    else:
        index = NearestNeighbors(
            n_neighbors=min(100, len(combined_features)),
            metric='cosine',
            algorithm='brute'
        )
        index.fit(combined_features)
        
        joblib.dump(index, 'music_index.pkl')
        # print("saved sklearn index to music_index.pkl")

    index_data = {
        'metadata': metadata,
        'deezer_features': deezer_features,
        'preview_embeddings': preview_embeddings,
        'combined_features': combined_features,
        'feature_keys': feature_keys,
        'embedding_dim': emb_dim,
        'has_faiss': HAS_FAISS
    }
    
    joblib.dump(index_data, 'index_data.pkl')
    # print("Saved index data to index_data.pkl")
    
    with_preview = sum(1 for e in preview_embeddings if e is not None)
    
    return index, index_data

if __name__ == '__main__':
    import sys
    
    max_previews = 500
    if len(sys.argv) > 1:
        max_previews = int(sys.argv[1])
    
    print(f"Building index with up to {max_previews} preview embeddings...")
    build_index(max_previews=max_previews)