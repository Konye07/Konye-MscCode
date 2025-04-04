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
from konye_m_packages import number_preproc, lemmat_processing, prepare_for_modeling_with_glove, remove_dates,remove_phone_numbers  # type: ignore
import pickle
import nltk # type: ignore
import contractions # type: ignore

print("Importok pipa")
###### Ötödik preprocessing #####

# Számok szavakká alakítása + stopszavak kivevése #

test_df = pd.read_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/test.csv")
train_df = pd.read_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/train.csv")
independent_df = pd.read_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/prepareed_independent.csv")


print("Fájlok behívása pipa")

########## Nem jó a packages szám-function mert minden stopszót eltávolít, viszont nekem kell mivel az a modellem volt a legjobb ########
########## Ezért itt újra definiálom a függvényt ############

def number_preproc2(df, text_column, new_column):
    if not hasattr(number_preproc2, "inflect_engine"):
        number_preproc2.inflect_engine = inflect.engine()

    inflect_engine = number_preproc2.inflect_engine

    def num_to_words(match):
        num = match.group()
        return inflect_engine.number_to_words(num)

    def clean_text(text):
        try:
            # Dátum és telefonszám eltávolítása
            text = remove_dates(text)
            text = remove_phone_numbers(text)

            # Lowercase
            text = text.lower()

            ########## Fake képek kifejezések eltávolítása#####
            phrases_to_remove = ['featured image', 'photo by', 'getty images']
            pattern = r'\b(?:' + '|'.join(map(re.escape, phrases_to_remove)) + r')\b/?'
            text = re.sub(pattern, '', text)

            text = re.sub(r'\s+/|/\s+', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            ####################################################

            # Contractions (don't -> do not)
            text = contractions.fix(text)

            # HTML és URL eltávolítása
            text = re.sub(r'<.*?>', '', text)
            text = re.sub(r'http\S+|www\.\S+', '', text)

            # Számok helyettesítése angol szavakkal
            text = re.sub(r'\b\d+\b', num_to_words, text)

            # Mondat szegmentáció
            sentences = sent_tokenize(text)

            # Szó tokenizálás stopword eltávolítás nélkül
            tokenized_sentences = []
            for sentence in sentences:
                # Írásjelek eltávolítása
                sentence_clean = sentence.translate(str.maketrans('', '', string.punctuation))
                sentence_clean = re.sub(r'[^a-z0-9\s]', '', sentence_clean).strip()

                # Tokenizálás
                tokens = word_tokenize(sentence_clean)

                tokenized_sentences.append(tokens)

            return tokenized_sentences  # Listában tokenizált mondatok

        except Exception as e:
            print(f"Error processing text: {text}. Error: {e}")
            return []

    df = df.copy()
    df[new_column] = df[text_column].fillna("").apply(clean_text)
    return df

test_df = number_preproc2(test_df, text_column='text', new_column='batch1')
print("Első function test fájlon pipa")
print(test_df.head())
train_df = number_preproc2(train_df, text_column='text', new_column='batch1')
print("Első function train fájlon pipa")
independent_df = number_preproc2(independent_df, text_column='text', new_column='batch1')
print("Első function independent fájlon pipa")

print("Első function lefutása pipa")

# Ellenőrzés hogy néz ki jelenleg

print(train_df.head())
print(test_df.head())

print("51. sorig lefutott")

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


train_df2.to_csv('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/05fifth_method/train_otodik.csv', index=False)
test_df2.to_csv('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/05fifth_method/teszt_otodik.csv', index=False)
independent_df2.to_csv('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/05fifth_method/indep_otodik.csv', index=False)

print("Első két preprocessing módszer lefuttatva, kimentve. 142. sor a kódban lefutott.")

### Model preparáló function ###

train_df2 = pd.read_csv('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/05fifth_method/train_otodik.csv')
test_df2 = pd.read_csv('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/05fifth_method/teszt_otodik.csv')
independent_df2 = pd.read_csv('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/05fifth_method/indep_otodik.csv')


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
padded_train05, embedding_matrix, tokenizer = prepare_for_modeling_with_glove(
    tokenized_train, glove_file,
    fit_tokenizer=True,
    max_vocab_size=MAX_VOCAB_SIZE,
    max_length=MAX_LENGTH,
    embedding_dim=EMBEDDING_DIM
)

padded_test05, _, _ = prepare_for_modeling_with_glove(
    tokenized_test, glove_file,
    tokenizer=tokenizer,
    fit_tokenizer=False,
    max_vocab_size=MAX_VOCAB_SIZE,
    max_length=MAX_LENGTH,
    embedding_dim=EMBEDDING_DIM
)

padded_independent05, _, _ = prepare_for_modeling_with_glove(
    tokenized_independent, glove_file,
    tokenizer=tokenizer,
    fit_tokenizer=False,
    max_vocab_size=MAX_VOCAB_SIZE,
    max_length=MAX_LENGTH,
    embedding_dim=EMBEDDING_DIM
)
 
# Mentés
np.save('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/05fifth_method/padded_train05.npy', padded_train05)
print("Harmadik function train fájlon pipa")
np.save('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/05fifth_method/padded_test05.npy', padded_test05)
print("Harmadik function test fájlon pipa")
np.save('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/05fifth_method/padded_independent05.npy', padded_independent05)
print("Harmadik function independent fájlon pipa")
np.save('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/05fifth_method/embedding_matrix.npy', embedding_matrix)
print("Embedding fájl pipa")

# Tokenizer mentése
with open('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/05fifth_method/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
