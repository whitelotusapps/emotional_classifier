import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from nltk.stem import PorterStemmer
from tqdm import tqdm
import concurrent.futures
import spacy
import re
import joblib
import time
import glob
import os
from datetime import datetime, timedelta
import json

desired_columns = "all"
#desired_columns = ["GEW"]

start_time = datetime.now()
os.system('clear')

def parallel_knn_models_creation(col, words):
    return classify_words_parallel(words, col)

def time_difference(start_time, stop_time):
    diff = stop_time - start_time
    minutes = int(diff.total_seconds() / 60)
    print(f"Transcription time: {minutes} minutes")

def get_most_recent_file(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    most_recent_file = max(files, key=os.path.getmtime)
    return most_recent_file

def expand_contractions(text):
    contractions_dict = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "I'd": "I would",
        "I'd've": "I would have",
        "I'll": "I will",
        "I'll've": "I will have",
        "I'm": "I am",
        "I've": "I have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so is",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have",}

    contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

    def replace(match):
        return contractions_dict[match.group(0)]

    return contractions_re.sub(replace, text)

def lookup_word_values(word):
    stemmed_word = ps.stem(word)
    
    row = word_df[word_df['Word'].apply(lambda x: ps.stem(x)) == stemmed_word]
    if not row.empty:
        return row[['V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum']].apply(pd.to_numeric, errors='coerce').values[0]
    return None

def classify_words_parallel(category_words, col):
    print(f"Generating model for: {col}")
    category_words = [str(word).strip().lower() for word in category_words]
    category_words = [word for word in category_words if word.isalpha()]

    category_word_values = {word: lookup_word_values(word) for word in category_words if lookup_word_values(word) is not None}

    category_word_matrix = np.array([v for v in category_word_values.values()])
    category_word_list = list(category_word_values.keys())

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    category_word_matrix = imputer.fit_transform(category_word_matrix)

    knn = NearestNeighbors(n_neighbors=2)
    knn.fit(category_word_matrix)
    print(f'Saving model: {col}_knn_model.pkl')
    joblib.dump(knn, f'{col}_knn_model.pkl')

    classification_results = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(lookup_word_values, word): word for word in words_to_classify}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing column {col}"):
            word = futures[future]
            word_values = future.result()

            if word_values is None:
                classification_results[word] = 0
                continue

            word_values = imputer.transform([word_values])

            distances, indices = knn.kneighbors(word_values)
            nearest_categories = [category_word_list[index] for index in indices[0]]
            assigned_category = nearest_categories[0]
            classification_results[word] = assigned_category

    return classification_results

print("Opening my_words.txt")
with open('my_words.txt', 'r') as file:
    my_words = [word.strip().lower() for word in file.readlines()]
    my_words.sort()

print("Reading list_of_emotional_thought.csv")
csv_file_name = 'list_of_emotional_thought.csv'
if desired_columns == "all":
    emotion_df = pd.read_csv(csv_file_name)
else:
    emotion_df = pd.read_csv(csv_file_name, usecols=lambda column: column in desired_columns)

most_recent_df = pd.DataFrame()

classified_words = set()  # Initialize classified_words as empty set

most_recent_file = get_most_recent_file('all_together_output_csv_file_*.csv')
if most_recent_file:
    most_recent_df = pd.read_csv(most_recent_file)
    classified_words = set(most_recent_df['idiolect'].values)
    print(f"Loaded the most recent file: {most_recent_file}")
else:
    print("No matching files found.")
    # Define the required columns
    required_columns = ['idiolect', 'part_of_speech'] + list(emotion_df.columns)
    # Initialize most_recent_df
    most_recent_df = pd.DataFrame(columns=required_columns)



unique_original_lines = set()
unique_expanded_lines = set()

print("Expanding contractions...")
for word in my_words:
    original = word.strip()
    expanded = expand_contractions(original)
    unique_original_lines.add(original)
    unique_expanded_lines.add(expanded)

words_to_classify = set(unique_original_lines) | set(unique_expanded_lines)

#print("BEFORE\n")
#print(f'emotion columns:\n{emotion_df.columns}\n')
#print(f'most recent columns:\n{most_recent_df.columns}')
#input("\n\nHERE\n\n")

#if not most_recent_df.empty:
emotion_df_columns = set(emotion_df.columns)
most_recent_df_columns = set(most_recent_df.columns)

#print("AFTER\n")
#print(f'emotion columns:\n{emotion_df.columns}\n')
#print(f'most recent columns:\n{most_recent_df.columns}')
#input("\n\nHERE\n\n")

new_columns = list(emotion_df_columns - most_recent_df_columns)
if new_columns:
    for col in new_columns:
        most_recent_df[col] = 0
    words_to_classify = unique_original_lines | unique_expanded_lines
else:
    words_to_classify = words_to_classify - classified_words

#print(f"words to classify:\n{words_to_classify}")
#input("\n\nHERE\n\n")

if len(words_to_classify) == 0:
    print(f"{most_recent_file} is up to date, no need to run classifier.")

if len(words_to_classify) > 0:
    print(f"Classifying {len(words_to_classify)} words...")
    print("Reading in word weights...")
    word_df = pd.read_csv('./Ratings_Warriner_et_al.csv')
    word_df['Word'] = word_df['Word'].astype(str).str.lower()
    word_df = word_df[word_df['Word'].str.isalpha()]

    ps = PorterStemmer()

    column_names = emotion_df.columns
    number_of_columns = len(column_names)

    print(f"Classifying {number_of_columns} schools of thought...")

    if desired_columns == "all":
        columns_to_process = emotion_df.columns
    else:
        columns_to_process = desired_columns

    words_columns = {name: emotion_df[name].str.lower().tolist() for name in columns_to_process}

    #print(f"words_columns:\n{words_columns}")
    #input("\n\nHERE\n\n")

    print("Start parallel processing...")
    classification_results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        classification_results = list(
            tqdm(
                executor.map(parallel_knn_models_creation, columns_to_process, [words_columns[col] for col in columns_to_process]),
                total=len(columns_to_process),
                desc="Creating k-NN models",
            )
        )    

    classification_dict = {}
    for word in words_to_classify:
        word_results = {column_names[i]: classification_results[i][word] for i in range(len(column_names))}
        if word in classified_words:
            row_index = most_recent_df.index[most_recent_df['idiolect'] == word][0]
            for col, value in word_results.items():
                #if isinstance(value, tuple):
                if isinstance(value, tuple) and value[1] < 0.5:
                    most_recent_df.at[row_index, col] = value[0]
                else:
                    most_recent_df.at[row_index, col] = value
        classification_dict[word] = word_results

    unique_words = sorted(set(words_to_classify))

    nlp = spacy.load('en_core_web_sm')


    #print(f"classification_dict:\n{json.dumps(classification_dict, indent=4)}")
    #input("\n\nHERE\n\n")

    column_names = most_recent_df.columns
    emotion_df = pd.DataFrame(unique_words, columns=['idiolect'])

    #print(f"column_names:\n{column_names}")
    #print(f"emotion_df:\n{emotion_df}")
    #input("\n\nHERE\n\n")

    #question_words = ["are", "because", "can", "did", "does", "feeling", "have", "how", "if", "is", "maybe", "my", "or", "our", "remember", "should", "since", "the", "they", "this", "want", "was", "we", "when", "where", "which", "who", "why", "will", "you", "your"]

    def process_idiolect(row):
        idiolect = row['idiolect']
        tokens = nlp(idiolect)

        if len(tokens) == 1:
            row['part_of_speech'] = tokens[0].pos_
        else:
            row['part_of_speech'] = ','.join([token.pos_ for token in tokens])

        if row['idiolect'] in classification_dict:
            for key, value in classification_dict[row['idiolect']].items():
#                    if isinstance(value, tuple) and value[1] < 0.5:
                row[key] = value

        return row

    emotion_df = emotion_df.apply(process_idiolect, axis=1)
#    if not most_recent_df.empty:
    #print("HERE - MOST RECENT DATAFRAME IS NOT EMPTY!")
    #input("\n\nHERE\n\n")
    unclassified_emotion_df = emotion_df[~emotion_df['idiolect'].isin(classified_words)].copy()
    unclassified_emotion_df = unclassified_emotion_df.apply(process_idiolect, axis=1)
    
    #print(f"unclassified_emotion_df:\n{unclassified_emotion_df}")
    #input("\n\nHERE\n\n")

    emotion_df = pd.concat([most_recent_df, unclassified_emotion_df])

    #print(f"emotion_df:\n{emotion_df}")
    #input("\n\nHERE\n\n")

    column_order = most_recent_df.columns.tolist()
    missing_columns = [col for col in emotion_df.columns if col not in column_order]
    column_order.extend(missing_columns)
    emotion_df = emotion_df[column_order]
    emotion_df = emotion_df.sort_values(by="idiolect")

    #print(f"emotion_df:\n{emotion_df}")
    #input("\n\nHERE\n\n")
#    else:


    timestr = time.strftime("%Y%m%d-%H%M%S")
    emotion_df.to_csv(f'all_together_output_csv_file_{timestr}.csv', index=False, na_rep='')
    stop_time = datetime.now()
    time_difference(start_time, stop_time)
    print(f"That's {len(words_to_classify) * number_of_columns} total classifications that were performed.")