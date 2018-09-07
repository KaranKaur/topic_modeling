import os
import re
import glob
import string
import random
import pickle
from math import floor
from random import shuffle
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import gensim
from gensim import corpora
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF
import numpy as np
from nltk.util import ngrams
from gensim.models import LdaModel
from pprint import pprint


# In[79]:

def load_chat_data(path):
    chat = []
    for filename in glob.glob(os.path.join(path, '*.tsv')):
        for i in range(1000):
            with open(filename, 'r') as f:
                for line in f:
                    temp = line.split('\t')
                    data_line = temp[-1].rstrip('\n')
                    chat.append(data_line)
    #pickle the chat data and usernames
    with open('chats.pkl', 'wb') as f:
        pickle.dump(chat, f)
    
    with open('data.txt', 'w') as f:
        for item in chat:
            f.write("%s\n" %item)        
        
    return chat  


# In[80]:

#Data Pre-processing: 
stopset = set(stopwords.words('english'))
lemma = WordNetLemmatizer()
stop_words = ['dont', 'know','anyone', 'help', 'im', 'please', 'trying', 'need', 'use', 'ihv','enter', 'make', 'sure', 'somthing', 'like', 'hi', 'hey', 'guy','doesnt',              'cant', 'think', 'would', 'like', 'all', 'am', 'and','or', 'ok', 'have' ,'a',              'it' ,'you','someone','ask', 'one', 'try', 'want', 'work', 'thanks']
def clean_data(doc):
    doc = doc.replace("'", "")
    doc = doc.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    temp = tokenizer.tokenize(doc)
    token_stopset = [w for w in temp if not w in stopset]
    token_stopwords = [w for w in token_stopset if not w in stop_words]
    token = [word for word in token_stopwords if word.isalpha()]
    normalized = [lemma.lemmatize(x) for x in token]
    return normalized


# In[81]:

chat_data= load_chat_data('.../Data/4')


# In[4]:

"""Question 1: Find 10 most Popular Topics in the corpus. This can be done by finding the 
the most common 10 unigrams/bigrams/trigrams."""


# In[44]:

data_cleaned = []
with open('data.txt','r') as f:
    text = f.read()    
    data_cleaned = clean_data(text)    


# # In[45]:

# #Unigrams:
unigrams = ngrams(data_cleaned,1)
freq_dist = nltk.FreqDist(unigrams)
most_common10 = freq_dist.most_common(10)
print(most_common10)


# # In[46]:

bigrams = ngrams(data_cleaned,2)
freq_dist = nltk.FreqDist(bigrams)
most_common10 = freq_dist.most_common(10)
print(most_common10)


# In[ ]:

"""Question2: Topic Detector"""


# In[82]:

def topic_detector_LDA(data,num_topics, chunksize, passes, iterations,eval_every=None):
    dictionary = corpora.Dictionary(data)
    
    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    dictionary.filter_extremes(no_below=20, no_above=0.5)
    
    #vectorize the data
    corpus = [dictionary.doc2bow(doc) for doc in data] 
    temp = dictionary[0]
    id2word = dictionary.id2token
    model = LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize,                        alpha='auto', eta='auto',                        iterations=iterations, num_topics=num_topics,                        passes=passes, eval_every=eval_every)
    top_topics = model.top_topics(corpus)

    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('Average topic coherence: %.4f.' % avg_topic_coherence)
 
    pprint(top_topics)


# In[83]:

data_lda = [clean_data(text) for text in chat_data]
topic_finder = topic_detector_LDA(data_lda[:1000], num_topics=10, chunksize=1000,                                   passes=10, iterations= 400,eval_every=None)


# In[6]:

##load data for a single file:
def load_single_file(filename):
    chat_single = []
    username_single = []
    with open(filename, 'r') as f:
        for line in f:
            temp = line.split('\t')
            data_line = temp[-1].rstrip('\n')
            chat_single.append(data_line)
            username_single.append(temp[1:3])
    return chat_single, username_single


# # In[68]:

#Change file path/name as needed
file_path = '../Data/4/30.tsv'
chat_data_single,username_single  = load_chat_data(file_path)
chat_data_temp = [clean_data(text) for text in chat_data_single]
chat_data_single_cleaned = [w for w in chat_data_temp if w not in username_single]


# # In[73]:

##Method 2 for Topic Detection: NMF
def topic_detector_NMF(data, num_topics,**kwargs ):
        vectorizer = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False)
        data_vec = vectorizer.fit_transform(data)
        words = np.array(vectorizer.get_feature_names())

        topicfinder = NMF(num_topics,**kwargs).fit(data_vec)
        topics_t = topicfinder.components_
        topics_t /= topics_t.max(axis = 1).reshape((-1, 1)) 
        
        #find keywords for topics
        def find_topic_keywords(topic_list, threshold = 1e-2):
            keywords_idx = np.abs(topic_list) >= threshold
            keywords_pre = np.where(np.sign(topic_list) > 0, "", "^")[keywords_idx]
            keywords = " | ".join(map(lambda x: "".join(x), zip(keywords_pre, words[keywords_idx])))
            return keywords
        
        ##finding keywords for topics:
        keywords_topics = map(find_topic_keywords, topics_t)
        return "\n".join("Topic %i: %s" % (i, j) for i, j in enumerate(keywords_topics))


# In[74]:

print(topic_detector_NMF(chat_data_single_cleaned,2))

