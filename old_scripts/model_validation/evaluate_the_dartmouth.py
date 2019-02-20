import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#from __future__ import print_function
from time import time

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

import pandas as pd 
import numpy as np
import os
from scipy import stats
from scipy.stats.stats import pearsonr
import glob
import wikipedia as wiki
import sys
import re
from sklearn.feature_extraction import text 
import string
from string import digits
import nltk
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import hypertools as hyp
import seaborn as sns
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

base_dir = '/dartfs-hpc/rc/home/5/f002s75/evaluate_models' 
data_dir = '/dartfs-hpc/rc/home/5/f002s75/web_scraping' 
topic_dir = '/dartfs-hpc/rc/home/5/f002s75/topic_modeling' 


n_samples = 2000
n_features = 1000 # might be too big
# REDUCE TOPIC LIST
n_components = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 150, 200] # number of topics
#n_components = [45]
n_top_words = 20

def print_top_words(model, feature_names, n_top_words, output_dir):
    
    myfile = open(os.path.join(output_dir,'topics.txt'), 'w')

    for topic_idx, topic in enumerate(model.components_):
            message = "Topic #%d: " % topic_idx
            message += " ".join([feature_names[i]
                                 for i in topic.argsort()[:-n_top_words - 1:-1]])
            myfile.write("%s\n" % message)

    myfile.close()

#stop = set(stopwords.words('english'))
#exclude = set(string.punctuation) 
#lemma = WordNetLemmatizer()



#def clean(doc):
#    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
#    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
#    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split()) # figure out what this is doing
#    return normalized

#flist = glob.glob(os.path.join(data_dir,'the_dartmouth','*.txt'))

#data_samples = []

#for file in flist:
    
#    if os.stat(file).st_size != 0:
    
#        file_text = open(file) 
#        text_1 = file_text.read()

#        data_samples.append(text_1)
        
#        file_text.close()

#file = open('all_articles.txt', 'w')
#for data_sample in data_samples:
#    file.write("%s\n" % data_sample)


data_samples_full = [line.rstrip('\n') for line in open(os.path.join(topic_dir,'all_articles.txt'))]


porter = PorterStemmer()

#data_samples_clean = [clean(doc).split() for doc in data_samples_full]

data_samples = []

for doc in data_samples_full:
    
    s = doc.lower()
    s = re.sub(r'([^\s\w]|_)+', ' ', s)
    s = re.sub(r'\s+', ' ', s) # gets rid of tabs
    s = re.sub(r'[^A-Za-z0-9]+', ' ', s)

    tokens = word_tokenize(s)
    stemmed = [porter.stem(word) for word in tokens]
    s = " ".join(stemmed)

    data_samples.append(s)


my_additional_stop_words = ['ve', 'll', 'd', 'm', 'o', 're', 'y', 'said', 'like']

stop_words = text.ENGLISH_STOP_WORDS.union(my_additional_stop_words)

for n_component in n_components:

    # Use tf-idf features for NMF.
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, 
                                       max_features=None, # max_features=n_features
                                       stop_words=stop_words)
    tfidf = tfidf_vectorizer.fit_transform(data_samples)

    # Use tf (raw term count) features for LDA.
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                    max_features=None,
                                    stop_words=stop_words)
    tf = tf_vectorizer.fit_transform(data_samples)

    # Fit the NMF model
    nmf = NMF(n_components=n_component, random_state=1,
              beta_loss='kullback-leibler', solver='mu', max_iter=2000, alpha=.1,
              l1_ratio=.5).fit(tfidf)

    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    lda = LatentDirichletAllocation(n_topics=n_component, max_iter=20,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0).fit(tf)

    tf_feature_names = tf_vectorizer.get_feature_names()


    ## SAVE

    flist = glob.glob(os.path.join(base_dir,'test_data_3','*.txt'))

    one_convo = []
    topics = []

    for file in flist:

        # get file name
        topic = file.split('/')[-1].split('_')[0]
        
        data = pd.read_table(file)
        data.columns = ['transcript']

        for i in range(len(data)):

            doc = data['transcript'].values
            s = ''.join(doc)
            s = s.lower()
            s = re.sub(r'([^\s\w]|_)+', ' ', s)
            s = re.sub(r'\s+', ' ', s) # gets rid of tabs

            tokens = word_tokenize(s)
            stemmed = [porter.stem(word) for word in tokens]
            s = " ".join(stemmed)
            
        one_convo.append(s)
        topics.append(topic)
       
        # start with first element
        topic_matrix = nmf.transform(tfidf_vectorizer.transform([one_convo[0]]))
        topic_matrix = pd.DataFrame(topic_matrix)


        for turn in one_convo[1:]: # skip first one
            topic_weights = nmf.transform(tfidf_vectorizer.transform([turn]))
            topic_weights = pd.DataFrame(topic_weights)
            topic_matrix = topic_matrix.append(topic_weights, ignore_index=True) 


        topic_matrix['topic'] = topics


    topic_matrix.to_csv(os.path.join(base_dir,'output_3','NMF_dartmouth_topic{0}.csv'.format(n_component)))


    # LDA

    flist = glob.glob(os.path.join(base_dir,'test_data_3','*.txt'))

    one_convo = []
    topics = []

    for file in flist:


        # get file name
        topic = file.split('/')[-1].split('_')[0]
        
        data = pd.read_table(file)
        data.columns = ['transcript']

        for i in range(len(data)):

            doc = data['transcript'].values
            s = ''.join(doc)
            s = s.lower()
            s = re.sub(r'([^\s\w]|_)+', ' ', s)
            s = re.sub(r'\s+', ' ', s) # gets rid of tabs

            tokens = word_tokenize(s)
            stemmed = [porter.stem(word) for word in tokens]
            s = " ".join(stemmed) 
            
        one_convo.append(s)
        topics.append(topic)

        # start with first element
        topic_matrix = lda.transform(tf_vectorizer.transform([one_convo[0]]))
        topic_matrix = pd.DataFrame(topic_matrix)


        for turn in one_convo[1:]: # skip first one
            topic_weights = lda.transform(tf_vectorizer.transform([turn]))
            topic_weights = pd.DataFrame(topic_weights)
            topic_matrix = topic_matrix.append(topic_weights, ignore_index=True) 


        topic_matrix['topic'] = topics

        topic_matrix.to_csv(os.path.join(base_dir,'output_3','LDA_dartmouth_topic{0}.csv'.format(n_component)))

#print_top_words(lda, tf_feature_names, n_top_words,base_dir) # do this with winning model


## Save Figures

#flist = glob.glob(os.path.join(base_dir,'output','*.csv'))

#for file in flist:
    
#    file_name = file.split('/')[-1].split('.csv')[0]
    ###### CHANGE THESE
#    new_header = ['choir_1','choir_2','choir_3','choir_4','computer_science_1','computer_science_2','computer_science_3','computer_science_4','dining_hall_1','dining_hall_2','dining_hall_3','dining_hall_4','football_1','football_2','football_3','football_4','fraternity_1','fraternity_2','fraternity_3','fraternity_4','sorority_1','sorority_2','sorority_3','sorority_4','theatre_1','theatre_2','theatre_3','theatre_4']

#    data = pd.read_csv(file)
#    data.pop('Unnamed: 0')
#    data.pop('topic')

#    topics_transpose = data.transpose()
#    topics_transpose.columns = new_header

#    corr_matrix=topics_transpose.corr()

#    ax = sns.heatmap(corr_matrix, square=True, cmap="RdBu_r", vmin=-1, vmax=1)

#    ax.figure.savefig(os.path.join(base_dir,'output','heatmaps','{0}.png'.format(file_name)))

    #plt.gcf().clear()


#flist = glob.glob(os.path.join(base_dir,'output','*.csv'))

#for file in flist:
    
#    file_name = file.split('/')[-1].split('.csv')[0]

#    data = pd.read_csv(file)
#    data.pop('Unnamed: 0')
#    class_labels = data.pop('topic')
    
#    hyp.plot(data, '.', group=class_labels, legend=list(set(class_labels)), animate = False, 
#             save_path=os.path.join(base_dir,'output','hypertools','{0}.png'.format(file_name)))






