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


def preprocess_documents(doc, stemming):
    """Preprocesses documents by making everything lower-case and
    removing punctuation. Stemming is optional.

    Parameters
    ----------
    doc : string
        String to be pre-processed
    stemming : bool
        If TRUE, stemming will be applied. If FALSE, stemming
        will not be applied

    Returns
    -------
    string
        A new string that has been pre-processed (everything to
        lower case, punctuation removed, stemming optional).

    """
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
            'Clean THIS up.,!@ stemming',
            stemming=True) == 'clean thi up stem'
    assert preprocess_documents(
            'Clean THIS up.,!@ stemming',
            stemming=False) == 'clean this up stemming'


def flag_stemming(stemming):
    """Indicated whether or not stemming was applied so
    this information can be clear when naming output
    files.

    Parameters
    ----------
    stemming : bool
        If TRUE, stemming was applied. If FALSE, stemming
        was not applied

    Returns
    -------
    string
        If TRUE, output is 'with_stemming'.
        If FALSE, output is 'without_stemming'.

    """
    return '{}_stemming'.format(('without', 'with')[stemming])


def apply_topic_models(stemming):
    """Apply pre-trained topic models to a set of validation
    data (documents with *known* topics)

    Parameters
    ----------
    stemming : bool
        If TRUE, stemming will be applied during preprocessing.
        If FALSE, stemming will not be applied during preprocessing.

    Returns
    -------
    .csv files
        .csv files that contain topic vectors for each validation
        document. One .csv file for every pre-trained model.

    """
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
                            ",".join(data['transcript'].values),
                            stemming))
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

        stemming_info = flag_stemming(stemming)

        topic_vector_dir = os.path.join(base_dir,
                                        'topic_vectors_{}'.format(
                                            stemming_info))

        if not os.path.exists(topic_vector_dir):
            os.makedirs(topic_vector_dir)

        topic_matrix.to_csv(os.path.join(topic_vector_dir,
                                         '{0}.csv'.format(
                                            model_name)))

    return base_dir, stemming


def flatten_topic_vectors(base_dir, stemming):
    """Take .csv file of topic vectors. Run all pairwise correlations.
    Take those values and flatten them to run statistics.

    Parameters
    ----------
    base_dir : path
        Path to directory that contains .csv files with topic vectors
    stemming : bool
        If TRUE, stemming was applied during preprocessing.
        If FALSE, stemming was not applied during preprocessing.

    Returns
    -------
    .npy files
        .npy files that contain all pairwise correlation values of
        the topic vectors, flattend.

    """
    stemming_info = flag_stemming(stemming)

    model_list = glob.glob(os.path.join(base_dir,
                                        'topic_vectors_{}'.format(
                                            stemming_info), '*.csv'))

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

        topic_vector_flat_dir = os.path.join(
                                        base_dir,
                                        'topic_vectors_flat_{}'.format(
                                            stemming_info))

        if not os.path.exists(topic_vector_flat_dir):
            os.makedirs(topic_vector_flat_dir)

        np.save(os.path.join(
            topic_vector_flat_dir, '{0}').format(
            name), data_flat)


# Compare to an 'ideal' version
# Probably a better way to generate the ideal version in the first place

def compare_to_perfect_model_performance(base_dir):
    """Compare model performance to an 'idealized' model
    performance. Do this by correlating two flattened
    correlation matrices (one from an 'idealized' model
    and the other from one of the pre-trained models.)

    Parameters
    ----------
    base_dir : path
        Path to directory that contains necessary info

    Returns
    -------
    pandas dataframe
        pandas dataframe with two columns: 1) model name and
        2) correlation of that model with 'idealized' model
        performance.

    """
    stemming_info = flag_stemming(stemming)

    perfect_model = pd.read_csv(os.path.join(base_dir, 'perfect_model.csv'),
                                header=None)
    perfect_model = perfect_model.values.flatten()

    flist = glob.glob(os.path.join(
                                base_dir,
                                'topic_vectors_flat_{}'.format(
                                    stemming_info), '*.npy'))

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


def plot_model_comparisions(evaluate_wiki, stemming):
    """Compare model performance to an 'idealized' model
    performance. Do this by correlating two flattened
    correlation matrices (one from an 'idealized' model
    and the other from one of the pre-trained models.)

    Parameters
    ----------
    evaluate_wiki : pandas dataframe
        pandas dataframe with two columns: 1) model name and
        2) correlation of that model with 'idealized' model
        performance.
    stemming : bool
        If TRUE, stemming was applied during preprocessing.
        If FALSE, stemming was not applied during preprocessing.

    Returns
    -------
    .png file
        A figure that demonstrates model performance across
        all pre-trained models.

    """

    stemming_info = flag_stemming(stemming)

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
    plt.title(
        "Model Comparison (validation docuents {0} stemming applied)".format(
            ('do not have', 'have')[stemming]), fontsize=20)

    axes = plt.gca()
    axes.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig('model_evaluation_{}.png'.format(
                stemming_info), edgecolor='none', dpi=300)
    plt.close()


if __name__ == '__main__':
    # two versions -- one with stemming applied to the validation
    # documents and one without stemming applied.
    base_dir, stemming = apply_topic_models(stemming=True)
    flatten_topic_vectors(base_dir, stemming)
    compare_to_perfect_model_performance(base_dir)
    performance_df = compare_to_perfect_model_performance(base_dir)
    plot_model_comparisions(performance_df, stemming)

    base_dir, stemming = apply_topic_models(stemming=False)
    flatten_topic_vectors(base_dir, stemming)
    compare_to_perfect_model_performance(base_dir)
    performance_df = compare_to_perfect_model_performance(base_dir)
    plot_model_comparisions(performance_df, stemming)
