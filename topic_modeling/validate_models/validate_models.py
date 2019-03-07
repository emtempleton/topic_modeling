import pandas as pd
import os
import glob
import re
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


def preprocess_documents(doc, stemming=True):
    porter = PorterStemmer()
    s = doc.lower()
    s = re.sub(r'([^\s\w]|_)+', ' ', s)
    s = re.sub(r'\s+', ' ', s)  # gets rid of tabs
    s = re.sub(r'[^A-Za-z0-9]+', ' ', s)
    tokens = word_tokenize(s)
    if stemming:
        stemmed = [porter.stem(word) for word in tokens]
        s = " ".join(stemmed)
    return s


# need two different ways to set directory to pass
# Travis CI (because .travis file is called from
# different directory)
current_file = os.getcwd().split('/')[-1]

if current_file == 'validate_models':
    base_dir = os.getcwd()
    model_dir = os.path.join(
        os.path.normpath(os.getcwd() + os.sep + os.pardir),
        'train_models', 'models_pickles')
else:
    base_dir = os.path.join(
            os.getcwd(), 'topic_modeling', 'validate_models')
    model_dir = os.path.join(
        os.getcwd(), 'topic_modeling', 'train_models', 'models_pickles')

model_list = glob.glob(os.path.join(model_dir, '*ing.pkl'))

for topic_model in model_list:

    model_name = topic_model.split('/')[-1].split('.pkl')[0]
    model_tf_name = '{}_tf.pkl'.format(model_name)

    with open(topic_model, 'rb') as pickle_file:
        model = pickle.load(pickle_file)

    with open(os.path.join(model_dir, model_tf_name), 'rb') as pickle_file_tf:
        tf_vectorizer = pickle.load(pickle_file_tf)

    flist = glob.glob(os.path.join(base_dir, 'validation_data', '*.txt'))

    one_convo = []
    topics = []

    for file in flist:

        # get file name
        topic = file.split('/')[-1].split('_')[0]

        data = pd.read_table(file)
        data.columns = ['transcript']
        one_convo.append(preprocess_documents(
                        ",".join(data['transcript'].values)))
        topics.append(topic)

        # start with first element
        topic_matrix = model.transform(tf_vectorizer.transform([one_convo[0]]))
        topic_matrix = pd.DataFrame(topic_matrix)

        for turn in one_convo[1:]:  # skip first one
            topic_weights = model.transform(tf_vectorizer.transform([turn]))
            topic_weights = pd.DataFrame(topic_weights)
            topic_matrix = topic_matrix.append(topic_weights,
                                               ignore_index=True)

    topic_matrix['topic'] = topics

    topic_matrix.to_csv(os.path.join(base_dir, 'topic_vectors',
                                     '{0}.csv'.format(model_name)))
