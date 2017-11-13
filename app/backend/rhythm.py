
# coding: utf-8

# In[1]:


import json
import pypinyin
import numpy as np
from gensim.models import Word2Vec
import thulac


# ## Load

# In[2]:


print('Loading...')

characterFrequency = json.load(open('../../rhythm/singleCharacterFrequency.json', 'r'))
characters = list(characterFrequency.keys())
char_finals = [ pypinyin.pinyin(char, style=pypinyin.Style.FINALS)[0][0] for char in characters]
char_initials = [ pypinyin.pinyin(char, style=pypinyin.Style.INITIALS)[0][0] for char in characters]
tupleCharacterFrequency = json.load(open('../../rhythm/tupleCharacterFrequency.json', 'r'))
characterFrequency[''] = 0
for key in characterFrequency:
    characterFrequency[''] += characterFrequency[key]
characterFrequency.update(tupleCharacterFrequency)

with_prev_freq = [np.array(        [ float(characterFrequency.get(prev + char, 0)) / characterFrequency.get(prev, 1) for prev in characters ])
                 for char in characters]


print('Loading complete!')


# ## 2-gram

# In[3]:


def predict_phrase_2gram(line, num_view=20):
    '''Predict the most frequent phrases that satisfiy the rhythm

    line: the previous line, ex: ['i', 'ao']
    num_view: the number of phrases that returned
    '''
    finals = [li[0] for li in pypinyin.pinyin(line, style=pypinyin.Style.FINALS)]
    initials = [li[0] for li in pypinyin.pinyin(line, style=pypinyin.Style.INITIALS)]

    phrase_length = len(finals)
    vocab_size = len(characters)

    probability = np.zeros([phrase_length, vocab_size], dtype='float')
    path = np.zeros([phrase_length, vocab_size], dtype='int') - 1
    for idx, char in enumerate(characters):
        if char_finals[idx] == finals[0] and char_initials[idx] != initials[0]:
            probability[0][idx] = characterFrequency[char]

    for k in range(1, phrase_length):
        for idx, char in enumerate(characters):
            if char_finals[idx] == finals[k]:
                probability[k][idx] = np.max(probability[k - 1] * with_prev_freq[idx])
                if probability[k][idx] > 0:
                    path[k][idx] = np.argmax(probability[k - 1] * with_prev_freq[idx])

    def path2phrase(k, idx):
        phrase = ''
        while k >= 0:
            if idx == -1: return None
            phrase = characters[idx] + phrase
            idx = path[k][idx]
            k -= 1
        return phrase

    return [path2phrase(phrase_length - 1, idx)            for idx in np.argsort(probability[phrase_length - 1])[::-1][:num_view]            if probability[phrase_length - 1][idx] > 0]


# In[5]:


if __name__ == '__main__':
    print(predict_phrase_2gram('风景'))
    print(predict_phrase_2gram('灿烂'))
    print(predict_phrase_2gram('倚老卖老'))
    print(predict_phrase_2gram('心境高雅韵如风'))


# ## Meaning satisfying rhythm

# In[5]:


def all_rhyme(word_x, word_y):
    if len(word_x) == len(word_y):
        finals_x = [li[0] for li in pypinyin.pinyin(word_x, style=pypinyin.Style.FINALS)]
        initials_x = [li[0] for li in pypinyin.pinyin(word_x, style=pypinyin.Style.INITIALS)]
        finals_y = [li[0] for li in pypinyin.pinyin(word_y, style=pypinyin.Style.FINALS)]
        initials_y = [li[0] for li in pypinyin.pinyin(word_y, style=pypinyin.Style.INITIALS)]

        for i in range(len(word_x)):
            if finals_x[i] != finals_y[i] or (i == 0 and initials_x[i] == initials_y[i]):
                return False
        return True
    return False


# In[50]:


model = Word2Vec.load('../../rhythm/word2vec_model')
cut = thulac.thulac(seg_only=True)

def word2finals(word):
    return [li[0] for li in pypinyin.pinyin(word, style=pypinyin.Style.FINALS)]

finals2word = {}
for word in model.wv.vocab.keys():
    finals2word.setdefault(' '.join(word2finals(word)),[]).append(word)

def predict_phrase_embedding(line, num_view=20):
    words = list(zip(*cut.cut(line)))[0]
    d = {}
    for word in words:
        try:
            candidate_list = np.array(finals2word[' '.join(word2finals(word))])
            candidate_similarity = list(map(lambda candidate: model.similarity(candidate, word), candidate_list))
            d[word] = list(candidate_list[np.argsort(candidate_similarity)[::-1][:num_view]])[1:]
        except Exception:
            d[word] = []

    return d


# In[53]:


if __name__ == '__main__':
    print(predict_phrase_embedding('风景'))
    print(predict_phrase_embedding('灿烂'))
    print(predict_phrase_embedding('倚老卖老'))
    print(predict_phrase_embedding('心境高雅韵如风'))


# In[244]:


if __name__ == '__main__':
    while True:
        line = input("Please input pinyin:\n")
        result = predict_phrase_2gram(line)
        print(result)

