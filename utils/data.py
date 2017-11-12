import os, Lyrics
path_saved = os.path.join(os.path.dirname(Lyrics.__file__), 'saved')
path_lyrics_filtered = os.path.join(path_saved, 'lyrics_filtered.pkl')

import pickle, random

def read_filtered():
    with open(path_lyrics_filtered, 'rb') as f:
        lyrics = pickle.load(f)
    random.shuffle(lyrics)
    return ['\n'.join(lyric) for lyric in lyrics]
