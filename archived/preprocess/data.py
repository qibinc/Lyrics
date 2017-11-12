
# coding: utf-8

# In[8]:


import re
import pickle
import numpy as np
import preprocess


# In[9]:


limit = {
    'minq': 0,
    'maxq': 20,
    'mina': 2,
    'maxa': 20
}

UNK = 'unk'


# In[10]:


def load_raw_data():
    """Return titles and lyrics."""
    
    titles_lyrics = preprocess.load_from_pickle()
    if titles_lyrics == None:
        preprocess.preprocess()
        titles_lyrics = preprocess.load_from_pickle()
    return titles_lyrics


# In[11]:


def lyrics_without_timing():
    """Return a list of lyrics, pure text, no timing"""
    _, lyrics = load_raw_data()
#     del_timing = lambda s: re.sub('\[.*\]', '', s).strip()
#     lyrics = [[del_timing(sentence) for sentence in lyric if del_timing(sentence) != '']\
#              for lyric in lyrics]
    
    del_not_chinese = lambda s: re.sub(u'[^\u4E00-\u9FA5 ]', '', s).strip()
    lyrics = [[del_not_chinese(sentence) for sentence in lyric               if del_not_chinese(sentence) != ''] for lyric in lyrics]
    
    lyrics = [[sentence for sentence in lyric
               if '作词' not in sentence and '作曲' not in sentence\
               and '编曲' not in sentence and '词曲' not in sentence\
              ]for lyric in lyrics]

    lyrics = [lyric for lyric in lyrics if len(lyric) > 15]
    return lyrics


# In[12]:


def q_a_lines(lyrics):
    """2 lists of sentences. Question and answer, respectively."""
    q, a = [], []
    ori_len = 0
    for lyric in lyrics:
        ori_len += len(lyric)
        for i in range(len(lyric) - 1):
            qlen, alen = len(lyric[i]), len(lyric[i+1])
            if qlen >= limit['minq'] and qlen <= limit['maxq']            and alen >= limit['mina'] and alen <= limit['maxa']:
                q.append(lyric[i])
                a.append(lyric[i+1])
    
    print('Q & A filtered {0}%'.format(100*(ori_len - len(q))/ori_len))
    return q, a


# In[13]:


def tokenize_single(qlines, alines):
    """Transfrom lines into lists of single characters.
    
    To do: tokenize_word
    """
    qtokenized = [[character for character in sentence] for sentence in qlines]
    atokenized = [[character for character in sentence] for sentence in alines]
    return qtokenized, atokenized


# In[14]:


def character_frequency(lyrics, vocab_size=3000, show=False):
    """Analyze Characters frequence.
    
    In a list of list of sentences.
    Example: [["song1", "hello world", "end"], ["song2", "happy end"]]
    """
    import numpy as np
    import itertools
    from collections import Counter, defaultdict

    iter_characters = itertools.chain(*itertools.chain(*lyrics))
    frequency_list = Counter(iter_characters).most_common()
    character, freq = zip(*frequency_list)
    
    if show:
        import matplotlib.pyplot as plt
        get_ipython().magic('matplotlib inline')
        plt.ylabel('frequency(log)')
        plt.xlabel('rank')
        plt.plot(range(len(frequency_list)), np.log(freq))
        plt.show()
        print('100 Most frequent word: {0}'.format(word[:100]))
    return list(character[:vocab_size]), list(freq[:vocab_size])


# In[15]:


def index(tokenized, vocab_size=3000):
    """Make volcabulary and lookup dictionary"""
    word, freq = character_frequency(tokenized, vocab_size)
    
    index2word = ['_', UNK] + word
    word2index = dict([(w, i) for i, w in enumerate(index2word)])
    
    return index2word, word2index


# In[16]:


def pad_seq(line, w2idx, maxlen):
    """zero padding at the tail"""
    padded = []
    for word in line:
        if word in w2idx:
            padded.append(w2idx[word])
        else:
            padded.append(w2idx[UNK])
    return np.array(padded + [0] * (maxlen - len(padded)))


# In[28]:


def zero_padding(qtokenized, atokenized, w2idx):
    """tokenized word sequences to idx sequences"""
    num_lines = len(qtokenized)
    idx_q = np.zeros([num_lines, limit['maxq']], dtype=np.int32)
    idx_a = np.zeros([num_lines, limit['maxa']], dtype=np.int32)
    
    for i in range(num_lines):
        idx_q[i] = pad_seq(qtokenized[i], w2idx, limit['maxq'])
        idx_a[i] = pad_seq(atokenized[i], w2idx, limit['maxa'])
    
    return idx_q[:, ::-1], idx_a


# In[29]:


if __name__ == '__main__':
#     lyrics = lyrics_without_timing()
#     qlines, alines = q_a_lines(lyrics)
#     qtokenized, atokenized = tokenize_single(qlines, alines)
#     idx2w, w2idx = index(qtokenized + atokenized)
    idx_q, idx_a = zero_padding(qtokenized, atokenized, w2idx)
    np.save('idx_q.npy', idx_q)
    np.save('idx_a.npy', idx_a)
    metadata = {
        'w2idx' : w2idx,
        'idx2w' : idx2w
    }
    pickle.dump(metadata, open('metadata.pkl', 'wb'))

