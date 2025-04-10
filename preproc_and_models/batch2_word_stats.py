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
from konye_m_packages import number_preproc, lemmat_processing, prepare_for_modeling_with_glove # type: ignore
import pickle
import nltk # type: ignore

train_df1 = pd.read_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/first_method/train_elso.csv")
train_df2 = pd.read_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/second_method/train_masodik.csv")
train_df3 = pd.read_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/third_method/train_harmadik.csv")
train_df4 = pd.read_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/fourth_method/train_negyedik.csv")
train_df5 = pd.read_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/fifth_method/train_otodik.csv")


def summarize_article_word_counts(dataframe, column_name):
    df = dataframe
    # Szavak hosszának összegyűjtése minden cikkből
    word_counts = []
    for index, row in df.iterrows():
        try:
            article = eval(row[column_name])
            if isinstance(article, list):
                word_count = sum(len(sentence) for sentence in article)
                word_counts.append(word_count)
        except Exception as e:
            print(f"Hiba a {index}. sor feldolgozásánál: {e}")

    if not word_counts:
        print("Nem sikerült érvényes adatokat találni.")
        return None

    # Alapvető statisztikák kiszámítása
    word_stats = {
        "Átlagos szószám": np.mean(word_counts),
        "Medián": np.median(word_counts),
        "Min szószám": np.min(word_counts),
        "Max szószám": np.max(word_counts),
        "Szórás": np.std(word_counts),
        "Kvartilisek (25-50-75-90%)": np.percentile(word_counts, [25, 50, 75, 90])
    }

    # Táblázat létrehozása
    stats_df = pd.DataFrame.from_dict(word_stats, orient="index", columns=["Érték"])

    return stats_df

print(summarize_article_word_counts(train_df1, "batch2"))
print(summarize_article_word_counts(train_df2, "batch2"))
print(summarize_article_word_counts(train_df3, "batch2"))
print(summarize_article_word_counts(train_df4, "batch2"))
print(summarize_article_word_counts(train_df5, "batch2"))