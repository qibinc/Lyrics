
# coding: utf-8

# In[4]:


import data


# In[9]:


def character_frequency(lyrics, vocab_size=3000):
    """Analyze Characters frequence.
    
    In a list of list of sentences.
    Example: [["song1", "hello world", "end"], ["song2", "happy end"]]
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    from collections import Counter

    get_ipython().magic('matplotlib inline')
    iter_characters = itertools.chain(*itertools.chain(*lyrics))
    frequency_list = Counter(iter_characters).most_common()
    word, freq = zip(*frequency_list)
    
    plt.ylabel('frequency(log)')
    plt.xlabel('rank')
    plt.plot(range(len(frequency_list)), np.log(freq))
    plt.show()
    print('100 Most frequent word: {0}'.format(word[:100]))
    return word[:vocab_size], freq[:vocab_size]


# In[10]:


if __name__ == '__main__':
    lyrics = data.lyrics_without_timing()
    character_frequency(lyrics)

