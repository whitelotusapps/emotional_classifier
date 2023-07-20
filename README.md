# Emotional Thoughts Classifier

This project aims to classify a list of words or phrases (called "idiolects") by associating them with different emotional categories or "schools of thought", using a k-nearest neighbors (k-NN) model. The project uses several Python libraries, including pandas, numpy, sklearn, NLTK, tqdm, spaCy, and joblib.

You can read about this project, and the inspiration for it, on from my LinkedIn article:

[So, You Want to Be a Data Scientist, Huh? Pub: 3](https://www.linkedin.com/pulse/so-you-want-data-scientist-huh-pub-3-zack-olinger)

Or on Medium:

[So, You Want to Be a Data Scientist, Huh? Part 3](https://medium.com/@therealzackolinger/so-you-want-to-be-a-data-scientist-huh-part-3-855a3d23f009)

## Sample output

The `EXAMPLE - all_together_output_csv_file_20230720-120327.csv` contains sample outout. The below table is an example of what the output will look like.

|idiolect|part_of_speech|berkeley_27 |discrete_emotion_theory|robert_plutchik|aristotle|charles_darwin|GEW        |circumplex_model|panas_scales|trzo_test     |
|--------|--------------|------------|-----------------------|---------------|---------|--------------|-----------|----------------|------------|--------------|
|abandon |PROPN         |envy        |sadness                |sadness        |envy     |weeping       |loneliness |depressed       |distressed  |disappointment|
|able    |ADJ           |sympathy    |interest               |trust          |kindness |reflection    |contentment|contented       |attentive   |connection    |
|absolute|ADJ           |entrancement|shyness                |               |         |mediation     |involvement|contented       |active      |similar       |

NOTE: Column `trzo_test` is an example of my own classifications, which are located in the `list_of_emotional_thought.csv` file.
## Dependencies
Before you run the code, please ensure that you have the following dependencies installed:

* Python 3.10
* pandas==1.5.3
* numpy==1.23.5
* scikit-learn==1.1.3
* nltk==3.8.1
* tqdm==4.64.1
* spacy==3.5.1
* joblib==1.2.0

You can install these dependencies using pip:
```
pip install -r requirements.txt
```


## Usage

Run the code in your Python environment. 

The words contained within the `my_words.txt` are to be your word, or the words that you wish to have classified. 

The `list_of_emotional_thought.csv` contains the various emotional categories and the associated words; I have also included an example of my own categorization to show how you can add your own classifications.

It also checks for any existing processed data in the form of `all_together_output_csv_file_*.csv` files and compares it with the new data to classify only the new idiolects.

The script uses k-nearest neighbors model to classify each idiolect based on the distance to the emotional words in the multi-dimensional emotional space. The model is saved for each column or category of emotions for reusability.

After processing all the idiolects, the results are saved in a new CSV file with a timestamp in its name, in the following format: `all_together_output_csv_file_YYYYMMDD-HHMMSS.csv`.

Users can adjust the distance to the nearest neighbor by adjusting the 0.5 value defined on the below line of code within the script:

```
if isinstance(value, tuple) and value[1] < 0.5:
```

The lower the value, the more strict the decision will be, and may result in fewer classifications. Setting this value to 1 will basically make this IF statement irrelevant.

## Script Overview
The script starts by importing all the necessary modules and defining the desired columns to be processed.

The main functions in the script include:

* `parallel_knn_models_creation(col, words)`: Classifies a list of words under a given category (or column) in parallel.
* `time_difference(start_time, stop_time)`: Computes and prints the time taken for the script to run.
* `get_most_recent_file(pattern)`: Finds the most recent CSV file matching a given pattern.
* `expand_contractions(text)`: Expands contractions in a given string.
* `lookup_word_values(word)`: Looks up and returns the emotional value(s) of a given word.
* `classify_words_parallel(category_words, col)`: Classifies a list of words under a given category in parallel, and saves the trained k-NN model for the category.

Then, the script reads the list of idiolects from 'my_words.txt', expands any contractions, and stores the unique idiolects in a set. It also reads the emotional categories from 'list_of_emotional_thought.csv' and checks for any previously processed data.

If previously processed data is found, the script computes the difference between the idiolects in the new data and the already classified idiolects to get the set of idiolects to be classified.

Next, the script proceeds to classify each idiolect under each emotional category in parallel, by training a k-NN model for each category based on the emotional values of the category's words. The classification results are then added to a dictionary, which is used to update the data frame with the new classifications.

Finally, the script saves the updated data frame as a new CSV file and prints the total time taken for the classification process.

Please note that the classification of words might take some time depending on the size of your data.

## Emotional Word Lists

1. [Berkeley’s 27 States of Emotion](https://news.berkeley.edu/2017/09/06/27-emotions/)
2. [Discrete Emotional Theory](https://en.wikipedia.org/wiki/Discrete_emotion_theory)
3. [Charles Darwin and here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2781895) and [here](https://simple.wikipedia.org/wiki/List_of_emotions)
4. [Aristotle](https://simple.wikipedia.org/wiki/List_of_emotions)
5. [GEW](https://academic.oup.com/book/2214/chapter-abstract/142269657?redirectedFrom=fulltext)
6. [Robert Plutchik](https://simple.wikipedia.org/wiki/List_of_emotions)
7. [Circumplex Model](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2367156)
8. [Panas Scales](https://positivepsychology.com/positive-and-negative-affect-schedule-panas/)
9. My own classifications as an example of how to add your own classifications to the output
# XANEW: The Extended Affective Norms for English Words

This code makes use of the XANEW word list for the weight of the words. Specifially, the only weights use in this code are the mean sums of valence, arousal, and dominance; specifically these rows:

V.Mean.Sum
A.Mean.Sum
D.Mean.Sum

This repository serves as a secondary distribution of the emotional word ratings provided by:

*Warriner, A.B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence, arousal, and dominance for 13,915 English 
lemmas. Behavior Research Methods, 45, 1191-1207.* 

Originally, the dataset has been released on this [website](http://crr.ugent.be/archives/1003) under a  
[Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License](https://creativecommons.org/licenses/by-nc-nd/3.0/deed.en_US).
Additionally, we provide a fix train-dev-test split to increase reproducibility in machine learning experiments.