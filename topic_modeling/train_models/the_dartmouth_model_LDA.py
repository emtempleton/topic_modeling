import os
import glob
import re
import nltk
import string
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction import text
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

# General TODO
# 1. Turn this into a function with parameters to
# enter on the command line
# 2. Store pickles and top words with reasonable
# naming convention
# 3. Use a sample text file for testing


def clean(document):
    stop_free = " ".join(
        [i for i in document.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


def print_top_words(model, feature_names, n_top_words, num_topics, output_dir):
    myfile = open(os.path.join(
            output_dir, 'LDA_dartmouth_topic{0}'.format(num_topics)), 'w')
    for topic_idx, topic in enumerate(model.components_):
            message = "Topic #{}: ".format(topic_idx)
            message += " ".join(
                [feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]])
            myfile.write("{}\n".format(message))
    myfile.close()


# need two different ways to set directory to pass
# Travis CI (because .travis file is called from
# different directory)
current_file = os.getcwd().split('/')[-1]

if current_file == 'train_models':
    base_dir = os.getcwd()
    data_dir = os.path.join(
        os.path.normpath(os.getcwd() + os.sep + os.pardir),
        'scrape_training_data')
else:
    base_dir = os.path.join(
            os.getcwd(), 'topic_modeling', 'train_models')
    data_dir = os.path.join(
        os.getcwd(), 'topic_modeling', 'scrape_training_data')

# Put all text from articles into one file,
# if not already
if not os.path.exists("all_articles.txt"):

    flist = glob.glob(os.path.join(data_dir, 'the_dartmouth', '*.txt'))

    data_samples = []

    for file in flist:

        if os.stat(file).st_size != 0:

            file_text = open(file)
            text_1 = file_text.read()

            data_samples.append(text_1)

            file_text.close()

    file = open('all_articles.txt', 'w')
    for data_sample in data_samples:
        file.write("{}\n".format(data_sample))
    file.close()


data_samples_full = [line.rstrip('\n') for line in open('all_articles.txt')]

porter = PorterStemmer()

data_samples_clean = [clean(doc).split() for doc in data_samples_full]

data_samples = []

for doc in data_samples_full:
    s = doc.lower()
    s = re.sub(r'([^\s\w]|_)+', ' ', s)
    s = re.sub(r'\s+', ' ', s)  # gets rid of tabs
    s = re.sub(r'[^A-Za-z0-9]+', ' ', s)
    s = re.sub(r"\[([A-Za-z0-9_]+)\]", r'', s)  # get rid of words in brackets

    tokens = word_tokenize(s)
    stemmed = [porter.stem(word) for word in tokens]
    s = " ".join(stemmed)

    data_samples.append(s)

my_additional_stop_words = [
    've', 'll', 'd', 'm', 'o', 're', 'y', 'said',
    'like', 'crosstalk', 'inaudible']
stop_words = text.ENGLISH_STOP_WORDS.union(my_additional_stop_words)

n_samples = 2000
n_features = 1000
n_components = [10]  # number of topics
# n_components = [int(sys.argv[1])]
num_top_words = 20

for n_component in n_components:

    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                    max_features=None,
                                    stop_words=stop_words)
    tf = tf_vectorizer.fit_transform(data_samples)

    lda = LatentDirichletAllocation(n_topics=n_component, max_iter=20,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0).fit(tf)

    tf_feature_names = tf_vectorizer.get_feature_names()

    top_words_dir = os.path.join(base_dir, 'models_topics')
    pickle_dir = os.path.join(base_dir, 'models_pickles')

    if not os.path.exists(top_words_dir):
        os.makedirs(top_words_dir)

    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)

    print_top_words(
        lda, tf_feature_names, num_top_words, n_component, top_words_dir)

    pickle_filename = 'LDA_dartmouth_topic{0}.pkl'.format(n_component)
    with open(os.path.join(pickle_dir, pickle_filename), 'wb') as file:
        pickle.dump(lda, file)
    file.close()
