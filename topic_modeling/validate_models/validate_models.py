import pandas as pd
import os
import glob
import re
import nltk
import pickle
import numpy as np
import matplotlib.pyplot as plt
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


def apply_topic_models():
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

        with open(os.path.join(
                model_dir, model_tf_name), 'rb') as pickle_file_tf:
            tf_vectorizer = pickle.load(pickle_file_tf)

        flist = glob.glob(os.path.join(base_dir, 'validation_data', '*.txt'))

        one_convo = []
        topics = []

        for document in flist:

            # get file name
            topic = document.split('/')[-1].split('_')[0]

            data = pd.read_table(document)
            data.columns = ['transcript']
            one_convo.append(preprocess_documents(
                            ",".join(data['transcript'].values)))
            topics.append(topic)

            # start with first element
            topic_matrix = model.transform(
                            tf_vectorizer.transform([one_convo[0]]))
            topic_matrix = pd.DataFrame(topic_matrix)

            for turn in one_convo[1:]:  # skip first one
                topic_weights = model.transform(
                                tf_vectorizer.transform([turn]))
                topic_weights = pd.DataFrame(topic_weights)
                topic_matrix = topic_matrix.append(topic_weights,
                                                   ignore_index=True)

        topic_matrix['topic'] = topics

        topic_vector_dir = os.path.join(base_dir, 'topic_vectors')

        if not os.path.exists(topic_vector_dir):
            os.makedirs(topic_vector_dir)

        topic_matrix.to_csv(os.path.join(topic_vector_dir,
                                         '{0}.csv'.format(model_name)))

    return base_dir


def flatten_topic_vectors(base_dir):
    model_list = glob.glob(os.path.join(base_dir, 'topic_vectors', '*.csv'))

    for model in model_list:

        data = pd.read_csv(model)
        name = model.split('/')[-1].split('.csv')[0]
        new_header = [
                        'computer_science_1', 'computer_science_2',
                        'computer_science_3', 'computer_science_4',
                        'football_1', 'football_2', 'football_3',
                        'football_4', 'fraternities_and_sororities_1',
                        'fraternities_and_sororities_2',
                        'fraternities_and_sororities_3',
                        'fraternities_and_sororities_4', 'neuroscience_1',
                        'neuroscience_2', 'neuroscience_3', 'neuroscience_4',
                        'religion_1', 'religion_2', 'religion_3', 'religion_4',
                        'theatre_1', 'theatre_2', 'theatre_3', 'theatre_4'
                        ]

        data = data.sort_values(by=['topic'])
        data.pop('Unnamed: 0')
        data.pop('topic')

        topics_transpose = data.transpose()
        topics_transpose.columns = new_header

        corr_matrix = topics_transpose.corr()
        data_flat = corr_matrix.values.flatten()

        topic_vector_flat_dir = os.path.join(base_dir, 'topic_vectors_flat')

        if not os.path.exists(topic_vector_flat_dir):
            os.makedirs(topic_vector_flat_dir)

        np.save(os.path.join(
            topic_vector_flat_dir, '{0}').format(name), data_flat)


# Compare to an 'ideal' version
# Probably a better way to generate the ideal version in the first place

def compare_to_perfect_model_performance(base_dir):
    perfect_model = pd.read_csv(os.path.join(base_dir, 'perfect_model.csv'),
                                header=None)
    perfect_model = perfect_model.values.flatten()

    flist = glob.glob(os.path.join(base_dir, 'topic_vectors_flat', '*.npy'))

    evaluate_wiki = pd.DataFrame()
    evaluate_wiki = evaluate_wiki.fillna(0)  # with 0s rather than NaNs

    counter = 0

    for file in flist:
        test_data = np.load(file)

        name = file.split('/')[-1].split('.npy')[0]

        correlation_term = np.corrcoef(test_data, perfect_model)[0][1]

        evaluate_wiki.at[counter, 'model'] = name
        evaluate_wiki.at[counter, 'correlation'] = correlation_term

        counter = counter + 1

    return evaluate_wiki


def plot_model_comparisions(evaluate_wiki):
    plt.style.use('seaborn-white')
    plt.figure(figsize=(20, 10))
    plt.bar(evaluate_wiki['model'], evaluate_wiki['correlation'],
            color='mediumblue')
    plt.margins(x=0.005)
    plt.xticks(rotation='vertical', fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Model', fontsize=20)
    plt.ylabel('correlation with a perfect fit', fontsize=20)
    plt.grid(axis='y')

    axes = plt.gca()
    axes.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig('model_evaluation.png', edgecolor='none', dpi=300)


if __name__ == '__main__':
    base_dir = apply_topic_models()
    flatten_topic_vectors(base_dir)
    compare_to_perfect_model_performance(base_dir)
    performance_df = compare_to_perfect_model_performance(base_dir)
    plot_model_comparisions(performance_df)
