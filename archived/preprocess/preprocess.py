
# coding: utf-8

# In[3]:


import json, pickle
from hanziconv import HanziConv


# In[10]:


def read_from_crawler(filename='crawled.json', show=False):
    """Read songs from crawled.

    return titiles, lyrics.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            songs = json.load(f)
            print('Read {0} lyrics from crawled.'.format(len(songs)))
            print('Example:')
            print(songs[0]['title'])
            print(songs[0]['lyric'])
            titles = [song['title'] for song in songs]
            lyrics = [song['lyric'] for song in songs]
            return titles, lyrics
    except Exception:
        print('crawled file not exist.')


# In[5]:


def dump_to_pickle(titles, lyrics):
    """Write titles and lyrics to pkl.
    """

    pickle.dump(titles, open('titles.pkl', 'wb'))
    pickle.dump(lyrics, open('lyrics.pkl', 'wb'))


# In[6]:


def filter_lyrics(titles, lyrics):
    """Filter out bad lyrics.
    """
    ori_len = len(titles)
    # filter out junk
    for i in range(len(lyrics)):
        lyrics[i] = lyrics[i].replace('\ufeff', '').replace('\r', '').replace('\n','')
        lyrics[i] = lyrics[i].split('[')[:-1]

    # filter out invalid lyrics
    titles = [title for title, lyric in zip(titles, lyrics) if lyric != [] and lyric[0] == '']
    lyrics = [lyric[1:] for lyric in lyrics if lyric != [] and lyric[0] == '']
    lyrics = [['[' + sentence for sentence in lyric if not 'lrcgc' in sentence]              for lyric in lyrics]
    
    # to Simplified Chinese
    
    lyrics = [[HanziConv.toSimplified(sentence)               for sentence in lyric] for lyric in lyrics]

    print('{0}% lyrics filtered.'.format(100 * (ori_len - len(titles)) / ori_len))
    return titles, lyrics


# In[7]:


def load_from_pickle():
    """Load tiltes and lyrics from pickle.
    """
    try:
        titles = pickle.load(open('titles.pkl', 'rb'))
    except Exception:
        print('titles.pkl not found.')
        return
    try:
        lyrics = pickle.load(open('lyrics.pkl', 'rb'))
    except Exception:
        print('titles.pkl not found.')
        return
    return titles, lyrics


# In[8]:


def preprocess():
    """Preprocess crawled data to pickle"""
    titles, lyrics = read_from_crawler(show=True)
    titles, lyrics = filter_lyrics(titles, lyrics)
    dump_to_pickle(titles, lyrics)


# In[11]:


if __name__ == '__main__':
    preprocess()

