## Project Overview:

I taught myself how to use python and then used it to cobble together scripts that allowed me to (i) scrape text from articles on the internet, (ii) train topic modeling using that text, (iii) validate and compare those models using performance on a task, and (iv) take that “winning” model and apply it to text data of interest.

Having now taken Psych 161 I am quite embarrassed by the topic modeling pipeline I’ve created. I plan to make incremental improvements on these scripts for my final project. 

## Specific Goals (I may not be able to do all of these in time for the final project)

1. Do a better job scraping text articles from the internet. I know that I haven’t figured out all the ways of dealing with punctuation and other formatting issues because some words still get smashed together, when there should be a space between them. I will write some simple tests to check that some of these common “uber words” aren’t created. 

2. The purpose of the model validation step is to compare models that vary across a set of parameters. I will create functions that allow the user to set these parameters (number of topics, text preprocessing steps, etc).

3. Currently, I create figures that summarize the results of model comparisons in a separate script. I would like these figures to run and output by default. I think I separated this initially because I was having issues creating figures in the cluster. I’ll add a parameter that will allow the user to turn off this option if they want (e.g., if it avoids issues on a cluster).

4. I want to figure out how to store a model (probably by pickling). Currently I re-train the “winning” model when I apply it to the data that I care about later. I would like to only have to train these models a single time, and store the ones that I want to use later.

5. There is currently a LOT of redundancy in these scripts. I will create functions for operations that I currently repeat.

6. Some of my paths are hard-coded. I will be more generalizable about calling path names.

7. I will make these scripts PEP8 compatible.   