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
from konye_m_packages import __all__ # type: ignore
from konye_m_packages import spacy_preproc, lemmat_processing, prepare_for_modeling_with_glove # type: ignore
import pickle
import nltk # type: ignore

print("Importok pipa")
###### Harmadik preprocessing #####

# spacy stopszó kivétele #

test_df = pd.read_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/test.csv")
train_df = pd.read_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/train.csv")
independent_df = pd.read_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/prepareed_independent.csv")


print("Fájlok behívása pipa")

test_df = spacy_preproc(test_df, text_column='text', new_column='batch1')
print("Első function test fájlon pipa")
train_df = spacy_preproc(train_df, text_column='text', new_column='batch1')
print("Első function train fájlon pipa")
independent_df = spacy_preproc(independent_df, text_column='text', new_column='batch1')
print("Első function independent fájlon pipa")

print("Első function lefutása pipa")

# Ellenőrzés hogy néz ki jelenleg

print(train_df.head())
print(test_df.head())

print("50. sorig lefutott")

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


train_df2.to_csv('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/03third_method/train_harmadik.csv', index=False)
test_df2.to_csv('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/03third_method/teszt_harmadik.csv', index=False)
independent_df2.to_csv('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/03third_method/indep_harmadik.csv', index=False)

print("Első két preprocessing módszer lefuttatva, kimentve. 74. sor a kódban lefutott.")

train_df2 = pd.read_csv('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/03third_method/train_harmadik.csv')
test_df2 = pd.read_csv('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/03third_method/teszt_harmadik.csv')
independent_df2 = pd.read_csv('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/03third_method/indep_harmadik.csv')

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
padded_train03, embedding_matrix, tokenizer = prepare_for_modeling_with_glove(
    tokenized_train, glove_file,
    fit_tokenizer=True,
    max_vocab_size=MAX_VOCAB_SIZE,
    max_length=MAX_LENGTH,
    embedding_dim=EMBEDDING_DIM
)

padded_test03, _, _ = prepare_for_modeling_with_glove(
    tokenized_test, glove_file,
    tokenizer=tokenizer,
    fit_tokenizer=False,
    max_vocab_size=MAX_VOCAB_SIZE,
    max_length=MAX_LENGTH,
    embedding_dim=EMBEDDING_DIM
)

padded_independent03, _, _ = prepare_for_modeling_with_glove(
    tokenized_independent, glove_file,
    tokenizer=tokenizer,
    fit_tokenizer=False,
    max_vocab_size=MAX_VOCAB_SIZE,
    max_length=MAX_LENGTH,
    embedding_dim=EMBEDDING_DIM
)

# Mentés
np.save('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/03third_method/padded_train03.npy', padded_train03)
print("Harmadik function train fájlon pipa")
np.save('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/03third_method/padded_test03.npy', padded_test03)
print("Harmadik function test fájlon pipa")
np.save('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/03third_method/padded_independent03.npy', padded_independent03)
print("Harmadik function independent fájlon pipa")
np.save('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/03third_method/embedding_matrix.npy', embedding_matrix)
print("Embedding fájl pipa")

# Tokenizer mentése
with open('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/03third_method/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
