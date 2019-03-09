# Overview

Together, these scripts allow you to (i) scrape text from articles on the internet, (ii) train topic modeling using that text, (iii) apply those models to a set of validation documents with *known* topics and (iv) compare models based on their similarity to an 'idealized' model that would have perfectly characterized the documents with known topics. 

# Motivation

When you try to train a topic model yourself, you’ll quickly realize that you need to make a lot of decisions and that those decisions don’t have a clear “right answer”. But, if you have a way to validate your model, you can explore the range of possible decisions and use this validation step as a way to compare model performance and help you decide which parameters make sense for you. 

This pipeline makes it easy to manipulate different parameters (number of topics, text pre-processing steps) and use an external "test" to pick the best performing model. This "winning" model can then be applied to documents of interest.

# Steps

Each step of this process is contained in a separate directory.

## 1. Scrape Training Data

./scrape_training_data

Contained in this directory are scripts that find, scrape, and store text from articles in The Dartmouth daily newspaer archives. 

## 2. Train Models

./train_models

The script 'train_models.py' takes the training data created in Step 1, pre-processes it, and uses it to train a series of topic models that vary across a set of parameters (number of topics, text pre-processing steps). These models are pickled and stored in 'models_pickles'. Each model also has a corresponding text files in 'models_topics' that lists the top words associated with topic. These collections of words allow the user to provide model 'labels' and to quickly assess the interpretability of the topics.

## 3. Validate Models

./validate_models

The script 'validate_models.py' applies each topic model created in Step 2 to a set of documents with *known* topics. These documents come from wikipedia pages. The text from each page is divided into 4 different documents -- each document contains 1/4 of the text from the original page. The validation step assesses the extent to which each topic model demonstrates that documents from the same wikipedia page are similar to each other. The performance of each model can be easily compared in the resulting figure. The 'winning' model can then be used in subsequent analyses. 

# Progress on this project (for Psych 161)

I cleaned up and streamlined these scripts as part of my final project in Psych 161. To be explicit about my progress, here is a list of improvements I made (in no particular order):

1. Got Travis CI up and running (and passing!).
2. Included tests for the first time (though still room for improvement / better coverage).
3. Ensured that all scripts are PEP8 compliant.
4. Made a master script that runs entire pipeline at once (useful for anyone starting from scratch or for leaving to run on the cluster).
5. Generally cleaned up existing scripts by making variable names more meaningful, removing redundancy, modularizing longer scripts (still room for improvement here), adding comments to functions.
6. Scripted the model comparision step (I was previously creating these correlation matrices and figures in an unruly jupyter notebook).
7. Fixed the issue of sentences being smooshed together when I scraped the articles.
8. Allowed for an important text preprocessing step (stemming) to be set as a parameter.
9. Finally stored trained models. I was previously re-running them at every step which was a huge waste of resources. I stored the models by pickling them, though I know Yarik has a preference for .json files. However, JSON serialization from sklearn pipelines is not straightforward (see one article here: https://cmry.github.io/notes/serialize) so I've left it as a pickle for now. 
10. No longer hardcoded paths (though will want to allow for users to set their own top level directories eventually).
11. Meta point: Actually worked with git! Now comfortable creating branches, commiting changes, submitting PRs, closing issues, and merging!
