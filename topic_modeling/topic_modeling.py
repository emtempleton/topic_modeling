import subprocess
import os

current_dir = os.getcwd().split('/')[-2:]

    if current_dir == ['topic_modeling', 'topic_modeling']:
        base_dir = os.getcwd()
    else:
        base_dir = os.path.join(
                os.getcwd(), 'topic_modeling')

base_dir = os.getcwd()

os.chdir(os.path.join(base_dir, 'scrape_training_data'))
subprocess.call(["python", "get_article_links.py"]) 
subprocess.call(["python", "scrape_articles.py"])

os.chdir(os.path.join(base_dir, 'train_models'))
subprocess.call(["python", "train_models.py"])

os.chdir(os.path.join(base_dir, 'validate_models'))
subprocess.call(["python", "validate_models.py"])
