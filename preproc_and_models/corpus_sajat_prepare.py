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
from konye_m_packages import preprocess_text, lemmat_processing, prepare_for_modeling_with_glove # type: ignore
import pickle
import nltk # type: ignore


##### Saját, 40 összegyűjtött cikk #####

sajat_gyujt = pd.read_excel("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/sajat_gyujtes.xlsx")

# Mennyi sora van
print(f"Eredeti sorok száma: {len(sajat_gyujt)}")

analyze_text_column(sajat_gyujt, "fake_news")
sajat_gyujt = sajat_gyujt[['label', 'text', 'text_length']]

sajat_gyujt = clean_dataset(sajat_gyujt)
print("Tisztítás lefuttatva")
print(sajat_gyujt.value_counts("label"))

sajat_gyujt.to_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/prepareed_sajat.csv", index=False)
