
from gensim import corpora
import gensim


def lda(bags, n_topics, iters=10):

    dictionary = corpora.Dictionary(bags)
    corpus = [dictionary.doc2bow(bag) for bag in bags]

    Lda = gensim.models.ldamulticore.LdaMulticore
    ldamodel = Lda(corpus, num_topics=n_topics, id2word=dictionary, passes=iters)

    return ldamodel


# In[84]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
import numpy as np

# In[107]:


# original version
def find_topic(texts, topic_model, n_topics, vec_model="tf", show_keywords=20, thr=1e-2, **kwargs):
        # 1. vectorization
        vectorizer = CountVectorizer() if vec_model== "tf" else TfidfVectorizer()
        text_vec = vectorizer.fit_transform(texts)

        words = np.array(vectorizer.get_feature_names())
        # 2. topic finding
        topic_models = {"nmf":NMF, "svd": TruncatedSVD, "lda":LatentDirichletAllocation, "kmeans":KMeans}
        topicfinder = topic_models[topic_model](n_topics, **kwargs).fit(text_vec)
        topic_dists = topicfinder.components_ if topic_model is not "kmeans" else topicfinder.cluster_centers_
        topic_dists /= topic_dists.max(axis = 1).reshape((-1,1))

        # 3. keywords for topics
        def _topic_keywords(topic_dist):
            keywords_index = np.abs(topic_dist) >= thr
            keywords_prefix = np.where(np.sign(topic_dist)>0, "","^")[keywords_index]
            keywords = " | ".join(map(lambda x: "".join(x), zip(keywords_prefix, words[keywords_index])))
            return keywords
        topic_keywords = map(_topic_keywords, topic_dists)
        return "\n".join("Topic %i:%s" % (i, t) for i, t in enumerate(topic_keywords))


# In[59]:


# vec model is tf

# print(find_topic(bag,"svd",20, vec_model = "tf"))
# print(find_topic(bag,"nmf",20, vec_model = "tf"))
# print(find_topic(bag,"lda",20, vec_model = "tf"))
# print(find_topic(bag,"kmeans",20, vec_model = "tf"))

# vec model is tfidf

#print(find_topic(bag,"svd",20, vec_model = "tfidf"))
# print(find_topic(bag, "nmf", 20, vec_model = "tfidf"))
#print(find_topic(bag,"lda",20, vec_model = "tfidf"))
# print(find_topic(bag,"kmeans",20, vec_model = "tfidf"))

