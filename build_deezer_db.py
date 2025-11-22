import requests
import time
import numpy as np
from pathlib import Path
import json
from audio_embeddings import extract_audio_features
import sys

PREVIEW_CACHE_DIR = Path('preview_cache')
PREVIEW_CACHE_DIR.mkdir(exist_ok=True)

def search_playlists(query, limit=10):
    url = f"https://api.deezer.com/search/playlist?q={query}&limit={limit}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()['data']

def get_playlist_tracks(playlist_id, limit=100):
    url = f"https://api.deezer.com/playlist/{playlist_id}"
    response = requests.get(url)
    response.raise_for_status()
    
    data = response.json()
    tracks = data.get('tracks', {}).get('data', [])
    
    track_list = []
    for track in tracks[:limit]:
        track_info = {
            'id': str(track['id']),
            'title': track['title'],
            'artist': track['artist']['name'],
            'album': track.get('album', {}).get('title', 'Unknown'),
            'duration': track.get('duration', 30),
            'preview_url': track.get('preview'),
        }
        track_list.append(track_info)
    
    return track_list

def get_chart_tracks(limit=100):
    url = f"https://api.deezer.com/chart/0/tracks?limit={limit}"
    response = requests.get(url)
    response.raise_for_status()
    
    data = response.json()
    tracks = data.get('data', [])
    
    track_list = []
    for track in tracks:
        track_info = {
            'id': str(track['id']),
            'title': track['title'],
            'artist': track['artist']['name'],
            'album': track.get('album', {}).get('title', 'Unknown'),
            'duration': track.get('duration', 30),
            'preview_url': track.get('preview'),
        }
        track_list.append(track_info)
    
    return track_list

def get_genre_tracks(genre_id, limit=50): 
    # artists
    url = f"https://api.deezer.com/genre/{genre_id}/artists?limit=20"
    response = requests.get(url)
    response.raise_for_status()
    
    artists = response.json().get('data', [])
    
    track_list = []
    for artist in artists[:10]:
        # top tracks (each artist)
        artist_url = f"https://api.deezer.com/artist/{artist['id']}/top?limit=5"
        artist_response = requests.get(artist_url)
        
        if artist_response.status_code == 200:
            tracks = artist_response.json().get('data', [])
            
            for track in tracks:
                if len(track_list) >= limit:
                    break
                    
                track_info = {
                    'id': str(track['id']),
                    'title': track['title'],
                    'artist': track['artist']['name'],
                    'album': track.get('album', {}).get('title', 'Unknown'),
                    'duration': track.get('duration', 30),
                    'preview_url': track.get('preview'),
                }
                track_list.append(track_info)
        
        time.sleep(0.1)
        
        if len(track_list) >= limit:
            break
    
    return track_list

def download_preview(url, track_id):
    if not url:
        return None
    
    cache_path = PREVIEW_CACHE_DIR / f"deezer_{track_id}.mp3"
    if cache_path.exists():
        return str(cache_path)
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        with open(cache_path, 'wb') as f:
            f.write(response.content)
        
        return str(cache_path)
    except Exception as e:
        print(f"failed to download {track_id}: {e}")
        return None

def build_database(source='chart', source_ids=None, max_tracks=500, output='deezer_db.npz'):
    all_tracks = []
    seen_ids = set()
    
    if source == 'chart':
        # print("\n1.Deezer chart")
        tracks = get_chart_tracks(limit=max_tracks)
        all_tracks.extend(tracks)
    
    elif source == 'playlist':
        # print(f"\n1.tracks from {len(source_ids)} playlists")
        for playlist_id in source_ids:
            print(f"   Playlist {playlist_id}...")
            try:
                tracks = get_playlist_tracks(playlist_id, limit=100)
                for t in tracks:
                    if t['id'] not in seen_ids and len(all_tracks) < max_tracks:
                        all_tracks.append(t)
                        seen_ids.add(t['id'])
                time.sleep(0.2)
            except Exception as e:
                print(f"   Error: {e}")
    
    elif source == 'genre':
        # print(f"\n1.tracks from {len(source_ids)} genres")
        for genre_id in source_ids:
            print(f"   Genre {genre_id}...")
            try:
                tracks = get_genre_tracks(genre_id, limit=50)
                for t in tracks:
                    if t['id'] not in seen_ids and len(all_tracks) < max_tracks:
                        all_tracks.append(t)
                        seen_ids.add(t['id'])
                time.sleep(0.2)
            except Exception as e:
                print(f"   Error: {e}")
    
    print(f"   Found {len(all_tracks)} tracks")

    tracks_with_previews = [t for t in all_tracks if t['preview_url']]
    # print(f"{len(tracks_with_previews)} have preview")
    
    if len(tracks_with_previews) == 0:
        return [], np.array([])

    # print(f"\n2. we download")
    
    valid_tracks = []
    feature_vectors = []
    feature_keys = ['tempo', 'spectral_centroid', 'spectral_rolloff', 
                    'zero_crossing_rate', 'rms_energy']
    
    for i, track in enumerate(tracks_with_previews):
        if i % 20 == 0:
            print(f"   Progress: {i}/{len(tracks_with_previews)}...")

        preview_path = download_preview(track['preview_url'], track['id'])
        if not preview_path:
            continue

        features = extract_audio_features(preview_path)
        if not features:
            continue

        vec = [float(features.get(key, 0)) for key in feature_keys]
        feature_vectors.append(vec)

        track_info = {
            'id': track['id'],
            'title': track['title'],
            'artist': track['artist'],
            'album': track['album'],
            'preview_url': track['preview_url'],
            'duration': track['duration'],
            'features': features,
            'source': 'deezer'
        }
        valid_tracks.append(track_info)
        
        time.sleep(0.05)
    
    print(f"{len(valid_tracks)} tracks done")
    
    if len(valid_tracks) == 0:
        #   print("\n no tracks?????????")
        return [], np.array([])
    
    feature_vectors = np.array(feature_vectors, dtype=np.float32)
    
    from sklearn.preprocessing import normalize
    feature_vectors = normalize(feature_vectors, axis=1)
    
    # print(f"feature shape: {feature_vectors.shape}")

    # print(f"\n3. saving to {output}")
    np.savez_compressed(
        output,
        features=feature_vectors,
        metadata=json.dumps(valid_tracks),
        feature_keys=feature_keys,
        source='deezer'
    )
    
    
    return valid_tracks, feature_vectors

POPULAR_PLAYLISTS = {
    'top': '3155776842',
    'hits': '1266975565',
    'pop': '1313621735',
    'rap': '1266974445',
    'rock': '1266974965',
    'electronic': '1266975085',
    'indie': '1313641735',
    'workout': '1313656735',
}

GENRE_IDS = {
    'pop': 132,
    'rock': 152,
    'rap': 116,
    'electronic': 113,
    'alternative': 85,
    'r&b': 165,
    'indie': 129,
    'metal': 464,
}

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='build db')
    parser.add_argument('--source', choices=['chart', 'playlist', 'genre'], 
                        default='chart', help='Source type')
    parser.add_argument('--ids', nargs='+', help='Playlist or genre IDs')
    parser.add_argument('--max-tracks', type=int, default=500, help='Max tracks')
    parser.add_argument('--output', default='deezer_db.npz', help='Output file')
    parser.add_argument('--list-playlists', action='store_true', 
                        help='List popular playlists')
    parser.add_argument('--list-genres', action='store_true',
                        help='List available genres')
    
    args = parser.parse_args()
    
    if args.list_playlists:
        for name, pid in POPULAR_PLAYLISTS.items():
            print(f"{name:15} → {pid}")
        sys.exit(0)
    
    if args.list_genres:
        for name, gid in GENRE_IDS.items():
            print(f"{name:15} → {gid}")
        sys.exit(0)
    
    if args.source in ['playlist', 'genre'] and not args.ids:
        if args.source == 'playlist':
            args.ids = [POPULAR_PLAYLISTS['hits'], POPULAR_PLAYLISTS['pop']]
            # print("default")
        else:
            args.ids = [GENRE_IDS['pop'], GENRE_IDS['rock']]
            # print("default")
    
    try:
        tracks, features = build_database(
            source=args.source,
            source_ids=args.ids,
            max_tracks=args.max_tracks,
            output=args.output
        )
        
    except Exception as e:
        print(f"\n{e}")
        import traceback
        traceback.print_exc()