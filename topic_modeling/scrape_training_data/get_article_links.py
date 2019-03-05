from bs4 import BeautifulSoup
import urllib.request
from time import sleep
from random import randint
from IPython.core.display import clear_output
from time import time

# change range to 0,50 when using cluster
pages = list(map(str, range(4)))

# change range to 2012,2019 when using cluster
year_urls = list(range(2015, 2016))

article_links = []

start_time = time()
requests = 0

for year_url in year_urls:

    for page in pages:

        html_page = urllib.request.urlopen(
            "http://www.thedartmouth.com/search?q=0&page={}&"
            "ti=0&tg=0&ty=0&ts_month=0&ts_day=0&ts_year={}&"
            "te_month=0&te_day=0&te_year={}&s=0&au=0&o=0&"
            "a=1".format(page, year_url, year_url+1))

        # Pause the loop
        sleep(randint(8, 15))

        # Monitor the requests
        requests += 1
        elapsed_time = time() - start_time
        print('Request:{}; Frequency: {} requests/s'.format(
            requests, requests / elapsed_time))
        clear_output(wait=True)

        soup = BeautifulSoup(html_page, 'html.parser')

        links = []
        for link in soup.find_all('a'):
            links.append(link.get('href'))

        for link in links:
            if link not in article_links:
                if 'article' in link.split('/'):
                    article_links.append(link)


file = open('article_links.txt', 'w')
for article_link in article_links:
    file.write("{}\n".format(article_link))
file.close()
