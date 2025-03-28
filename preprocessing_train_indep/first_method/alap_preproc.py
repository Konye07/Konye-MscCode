import pandas as pd
import re
import matplotlib.pyplot as plt
import ast
import string
from langdetect import detect
from collections import Counter
import numpy as np
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
import inflect
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konye_m_packages import __all__
from konye_m_packages import filtered_preproc, lemmat_pre, prepare_for_modeling_with_glove
import pickle


###### Első alappreprocessing #####

# Szakirodalom szerint filtered spacy stopszó #

test_df = pd.read_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/test.csv")
train_df = pd.read_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/train.csv")
independent_df = pd.read_csv("cpreprocessing_train_indep/databases/prepareed_independent.csv")


test_df = filtered_preproc(test_df, text_column='text', new_column='batch1')
train_df = filtered_preproc(train_df, text_column='text', new_column='batch1')
independent_df = filtered_preproc(independent_df, text_column='text', new_column='batch1')

nlp = spacy.load("en_core_web_sm")

## Itt látható miket veszek ki és marad bent
# Válogatott stopwords
filtered_stopwords = {
    word for word in nlp.Defaults.stop_words
    if nlp(word)[0].pos_ not in {"PRON", "ADV", "NOUN"}
}

# Kivett stopwords
removed_stopwords = {
    word for word in nlp.Defaults.stop_words
    if nlp(word)[0].pos_ in {"PRON", "ADV", "NOUN"}
}

print("Maradék stopwords amit removeol a szövegből:")
print(", ".join(sorted(filtered_stopwords)))
print("\nSzavak amik bent maradnak a szövegekben:")
print(", ".join(sorted(removed_stopwords)))


# Ellenőrzés hogy néz ki jelenleg

print(train_df.head())
print(test_df.head())

print("63. sorig lefutott")

# Második adag preproc, lemmatizálás

train_df2 = lemmat_pre(train_df, text_column='batch1', new_column='batch2')
test_df2 = lemmat_pre(test_df, text_column='batch1', new_column='batch2')
independent_df2 = lemmat_pre(independent_df, text_column='batch1', new_column='batch2')

test_df2.drop(columns=['text', 'batch1'], inplace=True)
train_df2.drop(columns=['text', 'batch1'], inplace=True)
independent_df2.drop(columns=['text', 'batch1'], inplace=True)

print(test_df2.head())
print(train_df2.head())
print(independent_df2.head())


train_df2.to_csv('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/first_method/train_elso.csv', index=False)
test_df2.to_csv('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/first_method/teszt_elso.csv', index=False)
independent_df2.to_csv('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/first_method/indep_elso.csv', index=False)

# print("Első két preprocessing módszer lefuttatva, kimentve. 84. sor a kódban lefutott.")


#### Statisztika a cikkekre hogy mekkora legyen a length #### 

df_all_batch2 = pd.concat([train_df2, test_df2, independent_df2], ignore_index=True)

# Szavak hosszának összegyűjtése minden cikkből
word_counts = []

for index, row in df_all_batch2.iterrows():
    article = eval(row["batch2"]) 
    if isinstance(article, list):
        word_count = sum(len(sentence) for sentence in article)
        word_counts.append(word_count)

# Alapvető statisztikák kiszámítása
word_stats = {
    "Átlagos szószám": np.mean(word_counts),
    "Medián": np.median(word_counts),
    "Min szószám": np.min(word_counts),
    "Max szószám": np.max(word_counts),
    "Szórás": np.std(word_counts),
    "Kvartilisek (25-50-75-90%)": np.percentile(word_counts, [25, 50, 75, 90])
}

# Statisztikák megjelenítése táblázatban
df_word_stats = pd.DataFrame.from_dict(word_stats, orient="index", columns=["Érték"])

# Eredmények kiírása
print(df_word_stats)


### Model preparáló function ###

# GloVe fájl elérési útja
glove_file = 'd:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/databases/glove.6B.300d.txt'

# Max szókincs és szekvencia hossz
MAX_VOCAB_SIZE = 25000
MAX_LENGTH = 700
EMBEDDING_DIM = 300  # GloVe 300d


# Szövegek előkészítése modellezéshez
tokenized_train = train_df2['batch2'].tolist()
tokenized_test = test_df2['batch2'].tolist()
tokenized_test = test_df2['batch2'].tolist()


# Train Tokenizer on training data
padded_train01, embedding_matrix, tokenizer = prepare_for_modeling_with_glove(tokenized_train, glove_file, fit_tokenizer=True)
padded_test01, _, _ = prepare_for_modeling_with_glove(tokenized_test, glove_file, tokenizer=tokenizer, fit_tokenizer=False)
padded_indepednent01, _, _ = prepare_for_modeling_with_glove(tokenized_test, glove_file, tokenizer=tokenizer, fit_tokenizer=False)

# Mentés
np.save('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/first_method/padded_train.npy', padded_train01)
np.save('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/first_method/padded_test.npy', padded_test01)
np.save('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/first_method/padded_test.npy', padded_indepednent01)
np.save('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/first_method/embedding_matrix.npy', embedding_matrix)

# Tokenizer mentése
with open('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/first_method/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
