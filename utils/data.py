import os
path_saved = os.path.join(os.path.dirname(__file__), '../saved')
path_lyrics_filtered = os.path.join(path_saved, 'lyrics_filtered.pkl')

import pickle, random

def read():
    """Read lyrics data from saved folder.

    :returns: a list of string

    .. code-block:: python

        ['天青色 等烟雨\\n而我在等你', '故事的小黄花\\n从出生那年就飘着']
    """
    with open(path_lyrics_filtered, 'rb') as f:
        lyrics = pickle.load(f)
    random.shuffle(lyrics)
    return ['\n'.join(lyric) for lyric in lyrics]
