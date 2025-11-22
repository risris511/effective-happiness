## Quick Start

### 1. Install Dependencies
```bash
pip install librosa numpy scikit-learn joblib requests soundfile
```

### 2. Build Database
```bash
# Option A: Use Deezer Chart (easiest)
python build_deezer_db.py --source chart --max-tracks 200

# Option B: Use specific playlists
python build_deezer_db.py --source playlist --ids 3155776842 1266975565

# Option C: Use genres
python build_deezer_db.py --source genre --ids 132 152
```

### 3. Build Index
```bash
python build_index.py
```

### 4. Query!
```bash
python query_script.py your_song.mp3
```

That's it! No API keys, no authentication, no hassle.

## Available Options

### List Available Playlists
```bash
python build_deezer_db.py --list-playlists
```

Output:
```
Popular Deezer Playlists:
  top             → 3155776842
  hits            → 1266975565
  pop             → 1313621735
  rap             → 1266974445
  rock            → 1266974965
  electronic      → 1266975085
  indie           → 1313641735
  workout         → 1313656735
```

### List Available Genres
```bash
python build_deezer_db.py --list-genres
```

Output:
```
Deezer Genres:
  pop             → 132
  rock            → 152
  rap             → 116
  electronic      → 113
  alternative     → 85
  r&b             → 165
```

## Usage Examples

### Build from Chart (Top 200 tracks)
```bash
python build_deezer_db.py --source chart --max-tracks 200
```

### Build from Multiple Playlists
```bash
python build_deezer_db.py \
  --source playlist \
  --ids 3155776842 1266975565 1313621735 \
  --max-tracks 500
```

### Build from Genres
```bash
python build_deezer_db.py \
  --source genre \
  --ids 132 152 116 \
  --max-tracks 300
```

### Custom Output File
```bash
python build_deezer_db.py \
  --source chart \
  --max-tracks 100 \
  --output my_music_db.npz
```

## Finding Playlist/Genre IDs
1. Go to https://www.deezer.com
2. Find a playlist you like
3. URL looks like: `https://www.deezer.com/playlist/3155776842`
4. Use the number: `3155776842`

## Troubleshooting

### "No tracks with previews"
- Some regions may have limited previews
- Try different playlists/genres
- Chart usually has best preview coverage

### "Connection timeout"
- Check internet connection
- Deezer API is usually very stable
- Try again in a few seconds

### "Preview download failed"
- Some preview URLs may be expired
- System automatically skips failed downloads
- 10-20% failure rate is normal

### Slow processing
- Each track takes ~1-2 seconds
- Use `--max-tracks` to limit size
- Preview downloads are cached