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
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

base_dir = '/dartfs-hpc/rc/home/5/f002s75/topic_modeling' 
data_dir = '/dartfs-hpc/rc/home/5/f002s75/web_scraping' 

n_samples = 2000
n_features = 1000 # might be too big
#n_components = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 150, 200] # number of topics
n_components = [int(sys.argv[1])]
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


data_samples_full = [line.rstrip('\n') for line in open('all_articles.txt')]

porter = PorterStemmer()

#data_samples_clean = [clean(doc).split() for doc in data_samples_full]

data_samples = []

for doc in data_samples_full:
    
    s = doc.lower()
    s = re.sub(r'([^\s\w]|_)+', ' ', s)
    s = re.sub(r'\s+', ' ', s) # gets rid of tabs
    s = re.sub(r'[^A-Za-z0-9]+', ' ', s)
    s = re.sub(r"\[([A-Za-z0-9_]+)\]", r'', s) # get rid of words in brackets

    tokens = word_tokenize(s)
    stemmed = [porter.stem(word) for word in tokens]
    s = " ".join(stemmed)

    data_samples.append(s)

my_additional_stop_words = ['ve', 'll', 'd', 'm', 'o', 're', 'y', 'said', 'like', 'crosstalk', 'inaudible']

stop_words = text.ENGLISH_STOP_WORDS.union(my_additional_stop_words)

for n_component in n_components:

    # Use tf (raw term count) features for LDA.
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                    max_features=None,
                                    stop_words=stop_words)
    tf = tf_vectorizer.fit_transform(data_samples)

    
    lda = LatentDirichletAllocation(n_topics=n_component, max_iter=20,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0).fit(tf)

    tf_feature_names = tf_vectorizer.get_feature_names()

    ## SAVE (1 turn = 1 document)

    window = 1

    output_dir_LDA = os.path.join(base_dir,'output','LDA_dartmouth_topic{0}_window_{1}'.format(n_component, window))

    
    if not os.path.exists(output_dir_LDA):
        os.makedirs(output_dir_LDA)
        

    # LDA

    flist = glob.glob(os.path.join(base_dir,'my_data','*.txt'))

    window = 1

    for file in flist:
        
        topic_matrix = pd.DataFrame() # reset dataframe

        one_convo = []

        # get file name
        name = file.split('/subs')[-1].split('.txt')[0]

        data = pd.read_csv(file, sep='\t', header=None)
        data.columns = ['timestamp', 'speaker', 'transcript']

        for i in range(len(data)):

            doc = data['transcript'][i]
            s = doc.lower()
            s = re.sub(r'([^\s\w]|_)+', ' ', s)
            s = re.sub(r'\s+', ' ', s) # gets rid of tabs
            s = re.sub(r"\[([A-Za-z0-9_]+)\]", r'', s) # get rid of words in brackets

            tokens = word_tokenize(s)
            stemmed = [porter.stem(word) for word in tokens]
            s = " ".join(stemmed) 
            one_convo.append(s)

        for turn in one_convo:
            topic_weights = lda.transform(tf_vectorizer.transform([turn]))
            topic_weights = pd.DataFrame(topic_weights)
            topic_matrix = topic_matrix.append(topic_weights, ignore_index=True) 


        topic_matrix['speaker'] = data['speaker']
        
        topic_matrix.to_csv(os.path.join(output_dir_LDA,'{0}.csv'.format(name)))

    	
    print_top_words(lda, tf_feature_names, n_top_words,output_dir_LDA)

    ## SAVE Sliding Versions

    # LDA

    flist = glob.glob(os.path.join(base_dir,'my_data','*.txt'))

    windows = [3, 5, 7, 9, 11]  # always an odd number
    
    for window in windows:

        turns_before = round((window - 1) / 2)
        turns_after = round(((window - 1) / 2) + 1)

        for file in flist:
            
            topic_matrix = pd.DataFrame() # reset dataframe

            one_convo = []

            # get file name
            name = file.split('/subs')[-1].split('.txt')[0]

            data = pd.read_csv(file, sep='\t', header=None)
            data.columns = ['timestamp', 'speaker', 'transcript']

            for i in range(len(data)):
                                
                if (i < turns_before):
                    
                    doc = data['transcript'][0:(i+turns_after)]
                    s = ''.join(doc)
                    s = s.lower()
                    s = re.sub(r'([^\s\w]|_)+', ' ', s)
                    s = re.sub(r'\s+', ' ', s) # gets rid of tabs
                    s = re.sub(r"\[([A-Za-z0-9_]+)\]", r'', s) # get rid of words in brackets

                    tokens = word_tokenize(s)
                    stemmed = [porter.stem(word) for word in tokens]
                    s = " ".join(stemmed)   
                    one_convo.append(s)
                    
                if (i > round((len(data) - turns_after))):

                    doc = data['transcript'][(i-turns_before):len(data)]
                    s = ''.join(doc)
                    s = s.lower()
                    s = re.sub(r'([^\s\w]|_)+', ' ', s)
                    s = re.sub(r'\s+', ' ', s) # gets rid of tabs
                    s = re.sub(r"\[([A-Za-z0-9_]+)\]", r'', s) # get rid of words in brackets

                    tokens = word_tokenize(s)
                    stemmed = [porter.stem(word) for word in tokens]
                    s = " ".join(stemmed)    
                    one_convo.append(s)
                    
                #else: # figure out this double if statement issue
                if (i >= turns_before) & (i <= round((len(data) - turns_after))):    
                    doc = data['transcript'][(i-turns_before):(i+turns_after)]
                    s = ''.join(doc)
                    s = s.lower()
                    s = re.sub(r'([^\s\w]|_)+', ' ', s)
                    s = re.sub(r'\s+', ' ', s) # gets rid of tabs
                    s = re.sub(r"\[([A-Za-z0-9_]+)\]", r'', s) # get rid of words in brackets

                    tokens = word_tokenize(s)
                    stemmed = [porter.stem(word) for word in tokens]
                    s = " ".join(stemmed) 
                    one_convo.append(s)

            
            for turn in one_convo:
                topic_weights = lda.transform(tf_vectorizer.transform([turn]))
                topic_weights = pd.DataFrame(topic_weights)
                topic_matrix = topic_matrix.append(topic_weights, ignore_index=True) 


            topic_matrix['speaker'] = data['speaker']

            output_dir_LDA = os.path.join(base_dir,'output','LDA_dartmouth_topic{0}_window{1}'.format(n_component, window))

            if not os.path.exists(output_dir_LDA):
                os.makedirs(output_dir_LDA)

            topic_matrix.to_csv(os.path.join(output_dir_LDA,'{0}.csv'.format(name)))

    # Separate by speaker

    # LDA

    flist = glob.glob(os.path.join(base_dir,'my_data','*.txt'))

    windows = [3, 5, 7, 9, 11] # always an odd number

    for window in windows:

        turns_before = round((window - 1) / 2)
        turns_after = round(((window - 1) / 2) + 1)

        for file in flist:

            # get file name
            name = file.split('/subs')[-1].split('.txt')[0]

            data = pd.read_csv(file, sep='\t', header=None)
            data.columns = ['timestamp', 'speaker', 'transcript']
            
            # S1
                
            topic_matrix_1 = pd.DataFrame() # reset dataframe

            one_convo_1 = []

            data_speaker_1 = data[data['speaker']=='S1']
            data_speaker_1 = data_speaker_1.reset_index(drop=True)

            for i in range(len(data_speaker_1)):

                # need when i == turns_before and i == turns_after

                if (i < turns_before):

                    doc = data_speaker_1['transcript'][0:(i+turns_after)]
                    s = ''.join(doc)
                    s = s.lower()
                    s = re.sub(r'([^\s\w]|_)+', ' ', s)
                    s = re.sub(r'\s+', ' ', s) # gets rid of tabs
                    s = re.sub(r"\[([A-Za-z0-9_]+)\]", r'', s) # get rid of words in brackets

                    tokens = word_tokenize(s)
                    stemmed = [porter.stem(word) for word in tokens]
                    s = " ".join(stemmed)     
                    one_convo_1.append(s)

                if (i > round((len(data_speaker_1) - turns_after))):

                    doc = data_speaker_1['transcript'][(i-turns_before):len(data_speaker_1)]
                    s = ''.join(doc)
                    s = s.lower()
                    s = re.sub(r'([^\s\w]|_)+', ' ', s)
                    s = re.sub(r'\s+', ' ', s) # gets rid of tabs
                    s = re.sub(r"\[([A-Za-z0-9_]+)\]", r'', s) # get rid of words in brackets

                    tokens = word_tokenize(s)
                    stemmed = [porter.stem(word) for word in tokens]
                    s = " ".join(stemmed)     
                    one_convo_1.append(s)

                #else: # figure out this double if statement issue
                if (i >= turns_before) & (i <= round((len(data_speaker_1) - turns_after))):    
                    doc = data_speaker_1['transcript'][(i-turns_before):(i+turns_after)]
                    s = ''.join(doc)
                    s = s.lower()
                    s = re.sub(r'([^\s\w]|_)+', ' ', s)
                    s = re.sub(r'\s+', ' ', s) # gets rid of tabs
                    s = re.sub(r"\[([A-Za-z0-9_]+)\]", r'', s) # get rid of words in brackets

                    tokens = word_tokenize(s)
                    stemmed = [porter.stem(word) for word in tokens]
                    s = " ".join(stemmed)     
                    one_convo_1.append(s)


            for turn in one_convo_1:
                topic_weights_1 = lda.transform(tf_vectorizer.transform([turn]))
                topic_weights_1 = pd.DataFrame(topic_weights_1)
                topic_matrix_1 = topic_matrix_1.append(topic_weights_1, ignore_index=True) 


            topic_matrix_1['speaker'] = data_speaker_1['speaker']
            
            # S2
                
            topic_matrix_2 = pd.DataFrame() # reset dataframe

            one_convo_2 = []

            data_speaker_2 = data[data['speaker']=='S2']
            data_speaker_2 = data_speaker_2.reset_index(drop=True)

            for i in range(len(data_speaker_2)):

                # need when i == turns_before and i == turns_after

                if (i < turns_before):

                    doc = data_speaker_2['transcript'][0:(i+turns_after)]
                    s = ''.join(doc)
                    s = s.lower()
                    s = re.sub(r'([^\s\w]|_)+', ' ', s)
                    s = re.sub(r'\s+', ' ', s) # gets rid of tabs
                    s = re.sub(r"\[([A-Za-z0-9_]+)\]", r'', s) # get rid of words in brackets

                    tokens = word_tokenize(s)
                    stemmed = [porter.stem(word) for word in tokens]
                    s = " ".join(stemmed)     
                    one_convo_2.append(s)

                if (i > round((len(data_speaker_2) - turns_after))):

                    doc = data_speaker_2['transcript'][(i-turns_before):len(data_speaker_2)]
                    s = ''.join(doc)
                    s = s.lower()
                    s = re.sub(r'([^\s\w]|_)+', ' ', s)
                    s = re.sub(r'\s+', ' ', s) # gets rid of tabs
                    s = re.sub(r"\[([A-Za-z0-9_]+)\]", r'', s) # get rid of words in brackets

                    tokens = word_tokenize(s)
                    stemmed = [porter.stem(word) for word in tokens]
                    s = " ".join(stemmed)    
                    one_convo_2.append(s)

                #else: # figure out this double if statement issue
                if (i >= turns_before) & (i <= round((len(data_speaker_2) - turns_after))):    
                    doc = data_speaker_2['transcript'][(i-turns_before):(i+turns_after)]
                    s = ''.join(doc)
                    s = s.lower()
                    s = re.sub(r'([^\s\w]|_)+', ' ', s)
                    s = re.sub(r'\s+', ' ', s) # gets rid of tabs
                    s = re.sub(r"\[([A-Za-z0-9_]+)\]", r'', s) # get rid of words in brackets

                    tokens = word_tokenize(s)
                    stemmed = [porter.stem(word) for word in tokens]
                    s = " ".join(stemmed)     
                    one_convo_2.append(s)


            for turn in one_convo_2:
                topic_weights_2 = lda.transform(tf_vectorizer.transform([turn]))
                topic_weights_2 = pd.DataFrame(topic_weights_2)
                topic_matrix_2 = topic_matrix_2.append(topic_weights_2, ignore_index=True) 


            topic_matrix_2['speaker'] = data_speaker_2['speaker']
            
            if data['speaker'][0] == 'S1':
                concat_df = pd.concat([topic_matrix_1,topic_matrix_2]).sort_index().reset_index(drop=True)
            if data['speaker'][0] == 'S2':
                concat_df = pd.concat([topic_matrix_2,topic_matrix_1]).sort_index().reset_index(drop=True)

            output_dir_LDA = os.path.join(base_dir,'output','LDA_dartmouth_topic{0}_window{1}_separate_speakers'.format(n_component, window))


            if not os.path.exists(output_dir_LDA):
                os.makedirs(output_dir_LDA)

            concat_df.to_csv(os.path.join(output_dir_LDA,'{0}.csv'.format(name)))