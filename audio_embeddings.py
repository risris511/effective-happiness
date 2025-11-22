import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore')
import soundfile as sf

def compute_mfcc_embedding(audio_path, sr=22050, n_mfcc=20):
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=30)
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        zero_crossing = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_mean = np.mean(mel_spec_db, axis=1)
        mel_std = np.std(mel_spec_db, axis=1)

        embedding = np.concatenate([
            mfcc_mean,
            mfcc_std,
            chroma_mean,
            chroma_std,
            [np.mean(spectral_centroid)],
            [np.std(spectral_centroid)],
            [np.mean(spectral_bandwidth)],
            [np.mean(spectral_rolloff)],
            np.mean(spectral_contrast, axis=1),
            [np.mean(zero_crossing)],
            [np.mean(rms)],
            mel_mean,
            mel_std
        ])
        
        return embedding
    
    except Exception as e:
        print(f"MFCC: {e}")
        return None

def compute_embedding(audio_path):
    return compute_mfcc_embedding(audio_path)

def extract_audio_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=22050, duration=30)
        
        tempo = librosa.beat.beat_track(y=y, sr=sr)
        
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        rms = np.mean(librosa.feature.rms(y=y))
        
        return {
            'tempo': float(tempo),
            'spectral_centroid': float(spectral_centroid),
            'spectral_rolloff': float(spectral_rolloff),
            'chroma': chroma_mean.tolist(),
            'zero_crossing_rate': float(zcr),
            'rms_energy': float(rms)
        }
    
    except Exception as e:
        print(f"features: {e}")
        return None

def compute_similarity_scores(query_features, candidate_features, query_deezer, candidate_deezer):
    scores = {}
    
    q_tempo = query_features.get('tempo', 120)
    c_tempo = candidate_features.get('tempo', 120)
    tempo_diff = abs(q_tempo - c_tempo)
    scores['rhythm'] = 1.0 / (1.0 + tempo_diff / 50.0)
    
    q_chroma = np.array(query_features.get('chroma', [0]*12))
    c_chroma = np.array(candidate_features.get('chroma', [0]*12))
    if q_chroma.any() and c_chroma.any():
        chroma_sim = np.dot(q_chroma, c_chroma) / (np.linalg.norm(q_chroma) * np.linalg.norm(c_chroma) + 1e-8)
        scores['harmony'] = float(chroma_sim)
    else:
        scores['harmony'] = 0.5
    
    q_sc = query_features.get('spectral_centroid', 0)
    c_sc = candidate_features.get('spectral_centroid', 0)
    q_zcr = query_features.get('zero_crossing_rate', 0)
    c_zcr = candidate_features.get('zero_crossing_rate', 0)
    
    sc_sim = 1.0 / (1.0 + abs(q_sc - c_sc) / 1000.0)
    zcr_sim = 1.0 / (1.0 + abs(q_zcr - c_zcr) * 100)
    scores['timbre'] = (sc_sim + zcr_sim) / 2.0
    
    if query_deezer and candidate_deezer:
        deezer_keys = ['danceability', 'energy', 'valence']
        diffs = []
        for key in deezer_keys:
            q_val = query_deezer.get(key, 0.5)
            c_val = candidate_deezer.get(key, 0.5)
            diffs.append(abs(q_val - c_val))
        scores['deezer_features'] = 1.0 - np.mean(diffs)
    else:
        scores['deezer_features'] = 0.5
    
    return scores

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        # print(f"computing embedding for: {audio_path}")
        
        emb = compute_embedding(audio_path)
        # if emb is not None:
            # print(f"embedding shape: {emb.shape}")
            # print(f"embedding stats: mean={emb.mean():.3f}, std={emb.std():.3f}")
        
        features = extract_audio_features(audio_path)
        # if features:
            # print(f"tempo: {features['tempo']:.1f} BPM")
            # print(f"spectral centroid: {features['spectral_centroid']:.1f}")
            # print(f"rms energy: {features['rms_energy']:.3f}")