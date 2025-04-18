import pandas as pd # type: ignore
import re
import matplotlib.pyplot as plt # type: ignore
import ast
import string
from langdetect import detect # type: ignore
from collections import Counter
import numpy as np # type: ignore
import spacy # type: ignore
from nltk.tokenize import sent_tokenize, word_tokenize # type: ignore
import inflect # type: ignore
from nltk import pos_tag # type: ignore
from nltk.stem import WordNetLemmatizer # type: ignore
from nltk.corpus import wordnet # type: ignore
from nltk.stem import PorterStemmer # type: ignore
import numpy as np # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import konye_m_packages # type: ignore
from konye_m_packages import __all__ # type: ignore
from konye_m_packages import filtered_preproc, lemmat_processing, prepare_for_modeling_with_glove # type: ignore
import pickle
import nltk # type: ignore

print("Importok pipa")
###### Első alappreprocessing #####

# Szakirodalom szerint filtered spacy stopszó #

test_df = pd.read_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/test.csv")
train_df = pd.read_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/train.csv")
independent_df = pd.read_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/prepareed_independent.csv")


print("Fájlok behívása pipa")

test_df = filtered_preproc(test_df, text_column='text', new_column='batch1')
print("Első function test fájlon pipa")
train_df = filtered_preproc(train_df, text_column='text', new_column='batch1')
print("Első function train fájlon pipa")
independent_df = filtered_preproc(independent_df, text_column='text', new_column='batch1')
print("Első function independent fájlon pipa")

print("Első function lefutása pipa")

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

print("71. sorig lefutott")

# Második adag preproc, lemmatizálás

train_df2 = lemmat_processing(train_df, text_column='batch1', new_column='batch2')
print("Második function train fájlon pipa")
test_df2 = lemmat_processing(test_df, text_column='batch1', new_column='batch2')
print("Második function test fájlon pipa")
independent_df2 = lemmat_processing(independent_df, text_column='batch1', new_column='batch2')
print("Második function independent fájlon pipa")

test_df2.drop(columns=['text', 'batch1'], inplace=True)
train_df2.drop(columns=['text', 'batch1'], inplace=True)
independent_df2.drop(columns=['text', 'batch1'], inplace=True)

print(test_df2.head())
print(train_df2.head())
print(independent_df2.head())


train_df2.to_csv('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/01first_method/train_elso.csv', index=False)
test_df2.to_csv('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/01first_method/teszt_elso.csv', index=False)
independent_df2.to_csv('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/01first_method/indep_elso.csv', index=False)

print("Első két preprocessing módszer lefuttatva, kimentve. 95. sor a kódban lefutott.")
print(independent_df2.head())

train_df2 = pd.read_csv('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/01first_method/train_elso.csv')
test_df2 = pd.read_csv('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/01first_method/teszt_elso.csv')
independent_df2 = pd.read_csv('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/01first_method/indep_elso.csv')

### Model preparáló function ###

# GloVe fájl elérési útja
glove_file = 'd:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/glove.6B.300d.txt'

# Max szókincs és szekvencia hossz
MAX_VOCAB_SIZE = 25000
MAX_LENGTH = 600
EMBEDDING_DIM = 300  # GloVe 300d


# Szövegek előkészítése modellezéshez
tokenized_train = train_df2['batch2'].tolist()
tokenized_test = test_df2['batch2'].tolist()
tokenized_independent = independent_df2['batch2'].tolist()


# Train Tokenizer on training data
padded_train01, embedding_matrix, tokenizer = prepare_for_modeling_with_glove(
    tokenized_train, glove_file,
    fit_tokenizer=True,
    max_vocab_size=MAX_VOCAB_SIZE,
    max_length=MAX_LENGTH,
    embedding_dim=EMBEDDING_DIM
)

padded_test01, _, _ = prepare_for_modeling_with_glove(
    tokenized_test, glove_file,
    tokenizer=tokenizer,
    fit_tokenizer=False,
    max_vocab_size=MAX_VOCAB_SIZE,
    max_length=MAX_LENGTH,
    embedding_dim=EMBEDDING_DIM
)

padded_independent01, _, _ = prepare_for_modeling_with_glove(
    tokenized_independent, glove_file,
    tokenizer=tokenizer,
    fit_tokenizer=False,
    max_vocab_size=MAX_VOCAB_SIZE,
    max_length=MAX_LENGTH,
    embedding_dim=EMBEDDING_DIM
)

# Mentés
np.save('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/01first_method/padded_train01.npy', padded_train01)
print("Harmadik function train fájlon pipa")
np.save('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/01first_method/padded_test01.npy', padded_test01)
print("Harmadik function test fájlon pipa")
np.save('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/01first_method/padded_independent01.npy', padded_independent01)
print("Harmadik function independent fájlon pipa")
np.save('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/01first_method/embedding_matrix.npy', embedding_matrix)
print("Embedding fájl pipa")

# Tokenizer mentése
with open('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/01first_method/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
