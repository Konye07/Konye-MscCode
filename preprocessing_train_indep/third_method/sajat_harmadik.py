from konye_m_packages import __all__ # type: ignore
from konye_m_packages import analyze_text_column, clean_dataset # type: ignore
import pandas as pd # type: ignore
from sklearn.utils import resample # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from langdetect import detect # type: ignore
import re
import nltk # type: ignore
nltk.download('words')
from nltk.corpus import words # type: ignore
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

sajat_df = pd.read_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/prepareed_sajat.csv")

sajat_df = spacy_preproc(sajat_df, text_column='text', new_column='batch1')

print("Preprocessing pipa")
print(sajat_df.value_counts("label"))

print(sajat_df.head())
print(sajat_df.head())

sajat_df2 = lemmat_processing(sajat_df, text_column='batch1', new_column='batch2')

print("Lemmatizálás rész pipa")

print(sajat_df2.head())

sajat_df2.to_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/third_method/own_harmadik.csv", index=False)

print("Saját cikkek kimentve")
print(sajat_df2.value_counts("label"))

### Model preparáló function ###
sajat_df2 = pd.read_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/third_method/own_harmadik.csv")

# GloVe fájl elérési útja
glove_file = 'd:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/glove.6B.300d.txt'

# Max szókincs és szekvencia hossz
MAX_VOCAB_SIZE = 25000
MAX_LENGTH = 600
EMBEDDING_DIM = 300  # GloVe 300d

sajat_df2 = pd.read_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/third_method/own_harmadik.csv")

# Szövegek előkészítése modellezéshez
tokenized_sajat = sajat_df2['batch2'].tolist()

# Tokenizer betöltése
with open('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/third_method/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Embedding mátrix betöltése
embedding_matrix = np.load('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/third_method/embedding_matrix.npy')

# Szövegek padding-elése a betöltött tokenizer-rel
padded_sajat, _, _ = prepare_for_modeling_with_glove(
    tokenized_sajat,
    glove_file=glove_file,
    tokenizer=tokenizer,
    fit_tokenizer=False,
    max_vocab_size=MAX_VOCAB_SIZE,
    max_length=MAX_LENGTH,
    embedding_dim=EMBEDDING_DIM
)

# Mentés
np.save('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/third_method/padded_sajat.npy', padded_sajat)
print("Saját szövegek előkészítve és elmentve.")