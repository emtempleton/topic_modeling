from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
from time import sleep
from random import randint
import os.path
import pandas as pd


def simple_get(url):
    """
    Attempts to get the content at `url` by making an HTTP GET request.
    If the content-type of response is some kind of HTML/XML, return the
    text content, otherwise return None
    """
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        return None


def is_good_response(resp):
    """
    Returns true if the response seems to be HTML, false otherwise
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200 
            and content_type is not None 
            and content_type.find('html') > -1)


def log_error(e):
    """
    It is always a good idea to log errors. 
    This function just prints them, but you can
    make it do anything.
    """
    print(e)


def test_simple_get():
    assert simple_get('https://www.google.com/') is not None


def test_log_error():
    assert log_error('This is a test') == print('This is a test')


base_dir = os.getcwd()

# make directory to hold articles
article_directory = os.path.join(base_dir, 'the_dartmouth')
if not os.path.exists(article_directory):
    os.makedirs(article_directory)

# read in article links
links_pandas = pd.read_csv(os.path.join(base_dir, 'article_links.txt'), header=None)
links = links_pandas[0]

for link in links:

    article_text = []
    raw_html = simple_get(link)

    title = link.split('/')[-1]
    
    if not os.path.exists(os.path.join(base_dir, 'the_dartmouth', '{0}.txt'.format(title))):

        html = BeautifulSoup(raw_html, 'html.parser')
        html = html.findAll('p', attrs={'class': None})

        for line_num, line_text in enumerate(html):
            article_text.append(line_text.text)

        paragraphs = article_text[2:]  # get rid of header info
        s = ' '.join(paragraphs)
        s = s.split('\n', 1)[0]

        if len(title) > 200:

            short_title = title[0:199]

            with open(os.path.join(base_dir, 'the_dartmouth', '{0}.txt'.format(short_title)), "wb") as text_file:
                    text_file.write(s.encode('utf8'))
        else:

            with open(os.path.join(base_dir, 'the_dartmouth', '{0}.txt'.format(title)), "wb") as text_file:
                    text_file.write(s.encode('utf8'))

        # Pause the loop
        sleep(randint(8, 15))
