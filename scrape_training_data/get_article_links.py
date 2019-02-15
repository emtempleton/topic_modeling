from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import re
import string
import glob
import os
import urllib.request
from time import sleep
from random import randint
from IPython.core.display import clear_output
from time import time
import os.path

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


pages = [str(i) for i in range(1,2)] #should go 1,50
years_url = [i for i in range(2015,2016)]

article_links = []

start_time = time()
requests = 0

for year_url in years_url:
    
    for page in pages:
        
        html_page = urllib.request.urlopen('http://www.thedartmouth.com/search?q=0&page=' + page +
                                           '&ti=0&tg=0&ty=0&ts_month=0&ts_day=0&ts_year=' + str(year_url) +
                                           '&te_month=0&te_day=0&te_year=' + str(year_url+1) + '&s=0&au=0&o=0&a=1')
        
        # Pause the loop
        sleep(randint(8,15))

        # Monitor the requests
        requests += 1
        elapsed_time = time() - start_time
        print('Request:{}; Frequency: {} requests/s'.format(requests, requests/elapsed_time))
        clear_output(wait = True)

        # Throw a warning for non-200 status codes
        #if response.status_code != 200:
        #    warn('Request: {}; Status code: {}'.format(requests, response.status_code))

        #Break the loop if the number of requests is greater than expected
        #if requests > 72:
        #    warn('Number of requests was greater than expected.')  
        #    break 
                                           
                                           
        soup = BeautifulSoup(html_page)
                                           
        links = []
        for link in soup.find_all('a'):
            links.append(link.get('href'))
                                           
        for link in links:
            try:
                if (link.split('/')[1] == 'article'):
                    article_links.append('http://www.thedartmouth.com{0}'.format(link))
            except IndexError:
                continue

file = open('article_links.txt', 'w')
for article_link in article_links:
    file.write("%s\n" % article_link)