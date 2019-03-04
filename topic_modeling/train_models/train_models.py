import os
import sys
import glob
import re
import nltk
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction import text
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


def print_top_words(model, feature_names, num_topics,
                    output_dir, n_top_words=20):
    myfile = open(os.path.join(
            output_dir, 'LDA_dartmouth_topic{0}'.format(num_topics)), 'w')
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #{}: ".format(topic_idx)
        message += " ".join(
            [feature_names[i]
                for i in topic.argsort()[:-n_top_words - 1:-1]])
        myfile.write("{}\n".format(message))
    myfile.close()


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


def test_preprocess_documents():
    assert preprocess_documents(
            'Clean THIS up.,!@ stemming') == 'clean thi up stem'
    assert preprocess_documents(
            'Clean THIS up.,!@ stemming',
            stemming=False) == 'clean this up stemming'


# will want an optional parameter for the documents.
# can set it here
def train_models(topics):

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

        all_articles = []

        for file in flist:

            if os.stat(file).st_size != 0:

                file_text = open(file)
                text_1 = file_text.read()

                all_articles.append(text_1)

                file_text.close()

        article_file = open('all_articles.txt', 'w')
        for article in all_articles:
            article_file.write("{}\n".format(article))
        article_file.close()

    all_articles_original = [line.rstrip('\n') for line in open(
                            'all_articles.txt')]

    all_articles_preprocessed = []

    for doc in all_articles_original:
        all_articles_preprocessed.append(preprocess_documents(doc))

    my_additional_stop_words = [
        've', 'll', 'd', 'm', 'o', 're', 'y', 'said',
        'like', 'crosstalk', 'inaudible']
    stop_words = text.ENGLISH_STOP_WORDS.union(my_additional_stop_words)

    for n_component in topics:

        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        max_features=None,
                                        stop_words=stop_words)
        tf = tf_vectorizer.fit_transform(all_articles_preprocessed)

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
            lda, tf_feature_names, n_component, top_words_dir)

        pickle_filename = 'LDA_dartmouth_topic{0}.pkl'.format(n_component)
        with open(os.path.join(
                    pickle_dir, pickle_filename), 'wb') as pickle_file:
            pickle_file.dump(lda, pickle_file)


if __name__ == '__main__':
    topics = [int(sys.argv[1])]
