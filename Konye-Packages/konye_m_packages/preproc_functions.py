# Eloszor az alap adatbazisra lefuttatott szukseges, preprocess elotti tisztitasok, statisztikak

import pandas as pd
import re
import matplotlib.pyplot as plt
import ast
import contractions
import string
from langdetect import detect
from collections import Counter
import numpy as np
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
import inflect
from collections import Counter
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Statisztikak dataframeekre

def analyze_text_column(df, name):
    print(f"\n===== {name} DataFrame =====")

    # Sorok szama
    row_count = df.shape[0]
    print(f"Sorok száma: {row_count}")

    # Szoveghosszak
    df['text_length'] = df['text'].astype(str).apply(len)

    # Statisztikai mutatok
    stats = df['text_length'].describe()

    print(f"\nStatisztikák a 'text' oszlop hosszára:")
    print(f"Minimális hossz: {stats['min']}")
    print(f"Maximális hossz: {stats['max']}")
    print(f"Átlagos hossz: {stats['mean']:.2f}")
    print(f"Medián hossz: {stats['50%']:.2f}")
    print(f"Szórás: {stats['std']:.2f}")
    print(f"1. kvartilis: {stats['25%']:.2f}")
    print(f"3. kvartilis: {stats['75%']:.2f}")

# stopszo nelkuli abra

def plot_most_common_words(df, name, top_n=15):
    nlp = spacy.load("en_core_web_sm")
    def clean_and_tokenize(text):
        doc = nlp(text.lower())  # Kisbetus, tokenizaas abrazolashoz
        return [token.text for token in doc if token.is_alpha and token.text not in nlp.Defaults.stop_words and token.text not in {"s", "t", "nt"}]

    # Szavak osszegyujtese
    words = []
    for text in df["text"].dropna():  # Üres ertekek kiszurese
        words.extend(clean_and_tokenize(text))

    # Gyakoriság számolás
    word_freq = Counter(words)
    most_common_words = word_freq.most_common(top_n)  # Leggyakoribb szavak kiválasztasa

    if not most_common_words:
        print(f"Nincs elegendő adat a {name} DataFrame-ben az ábrázoláshoz.")
        return

    words, counts = zip(*most_common_words)

    # Diagram
    plt.figure(figsize=(12, 6))
    plt.bar(words, counts, color='royalblue')
    plt.xlabel("Szavak")
    plt.ylabel("Előfordulás")
    plt.title(f"Leggyakoribb {top_n} szó a {name} DataFrame-ben")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


#stopszavakat tartalmazó ábra, ugyanaz csak stopszo szures kiveve
def plot_most_common_words2(df, name, top_n=15):
    nlp = spacy.load("en_core_web_sm")
    def clean_and_tokenize(text):
        doc = nlp(text.lower()) 
        return [token.text for token in doc if token.is_alpha]

    # Szavak
    words = []
    for text in df["text"].dropna():
        words.extend(clean_and_tokenize(text))

    # Gyakorisag
    word_freq = Counter(words)
    most_common_words = word_freq.most_common(top_n)

    if not most_common_words:
        print(f"Nincs elegendő adat a {name} DataFrame-ben az ábrázoláshoz.")
        return

    words, counts = zip(*most_common_words)

    # Diagram
    plt.figure(figsize=(12, 6))
    plt.bar(words, counts, color='royalblue')
    plt.xlabel("Szavak")
    plt.ylabel("Előfordulás")
    plt.title(f"Leggyakoribb {top_n} szó a {name} DataFrame-ben")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

# Alap tisztitas elvegzese, angolnyelv-ures-stb
# Nyelv felismero function
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# Fuggveny a tisztitashoz
def clean_dataset(dataframe):
    # Adatok
    df = dataframe

    # Eredeti sorok
    original_row_count = len(df)

    # Nem szukseges oszlop dropolasa
    df = df.drop('text_length', axis=1)

    # A 'text' oszlop hianyzo ertekeinek kitoltese ures karakterlanccal
    df['text'] = df['text'].fillna("")

    # Nyelv eszlelese
    df['language'] = df['text'].apply(detect_language)

    # Ures es nem angol sorok kiszurese
    df = df[(df['text'].str.strip() != "") & (df['language'] == 'en')]

    # A sorok szama
    cleaned_row_count = len(df)

    # Eltavolitott sorok szama
    rows_removed = original_row_count - cleaned_row_count

    # Osszegzes
    print(f"Eredeti sorok száma: {original_row_count}")
    print(f"Eltávolított sorok száma: {rows_removed}")
    print(f"Sorok száma tisztítás után: {cleaned_row_count}")

    # Ellenorzes
    remaining_non_english = df[df['language'] != 'en']
    remaining_empty = df[df['text'].str.strip() == ""]

    if not remaining_non_english.empty or not remaining_empty.empty:
        print("Nem sikerült kivenni mindent")
    else:
        print("Az összes nem angol és üres szöveges sor eltávolítotva.")

    return df

# datumok kivetele regexel
def remove_dates(text):
    iso_pattern = r'\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b'
    non_iso_pattern = r'\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b'
    text = re.sub(iso_pattern, '', text)
    text = re.sub(non_iso_pattern, '', text)
    return text.strip()

# telefonszamok kivetele regexel
def remove_phone_numbers(text):
    phone_pattern = r'\b(?:\+36\s?\d{1,2}|\(06[-\s]?\d{1,2}\)|06[-\s]?\d{1,2})[-.\s]?\d{3}[-.\s]?\d{4}\b'
    return re.sub(phone_pattern, '', text)

################################### 3 lepeses preprocessing ###############################################
#################### Elso lepes a zaj tisztitas, 4 kulonbozo modszer ehhez ####################

## Elso, szakirodalom szerint szűrt spacy stopszavak ##

def filtered_preproc(df, text_column, new_column):
    if not hasattr(filtered_preproc, "nlp"):
        filtered_preproc.nlp = spacy.load("en_core_web_sm")
        stopwords = filtered_preproc.nlp.Defaults.stop_words
        filtered_preproc.filtered_stopwords = {
            word for word in stopwords
            if filtered_preproc.nlp(word)[0].pos_ not in {"PRON", "ADV", "NOUN"}
        }

    filtered_stopwords = filtered_preproc.filtered_stopwords

    def clean_text(text):
        try:
            text = remove_dates(text)
            text = remove_phone_numbers(text)
            text = text.lower()

            phrases_to_remove = ['featured image', 'photo by', 'getty images']
            pattern = r'\b(?:' + '|'.join(map(re.escape, phrases_to_remove)) + r')\b/?'
            text = re.sub(pattern, '', text)
            text = re.sub(r'\s+/|/\s+', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()

            text = contractions.fix(text)
            text = re.sub(r'<.*?>', '', text)
            text = re.sub(r'http\S+|www\.\S+', '', text)
            text = re.sub(r'\b\d+\b', 'NUMTOKEN ', text)

            sentences = sent_tokenize(text)
            tokenized_sentences = []
            for sentence in sentences:
                sentence_clean = sentence.translate(str.maketrans('', '', string.punctuation))
                sentence_clean = re.sub(r'[^a-z0-9\s]', '', sentence_clean).strip()
                tokens = word_tokenize(sentence_clean)
                filtered_tokens = [
                    word for word in tokens
                    if word not in filtered_stopwords and word not in {"s", "t"}
                ]
                tokenized_sentences.append(filtered_tokens)

            return tokenized_sentences

        except Exception as e:
            print(f"Error processing text: {text}. Error: {e}")
            return []

    df = df.copy()
    df[new_column] = df[text_column].fillna("").apply(clean_text)
    return df


## Masodik, összes stopszó bentmarad ##

def preprocess_text(df, text_column, new_column):
    def clean_text(text):
        try:
            # Dátum és telefonszám eltávolítása
            text = remove_dates(text)
            text = remove_phone_numbers(text)

            # Lowercase
            text = text.lower()

            ########## Fake képek kifejezések eltávolítása#####
            phrases_to_remove = ['featured image', 'photo by', 'getty images']

            # A kifejezésekből regex minta készítése, figyelembe véve az opcionális perjelet és a szóhatárokat
            pattern = r'\b(?:' + '|'.join(map(re.escape, phrases_to_remove)) + r')\b/?'

            # A mintázat alkalmazása a szövegre
            text = re.sub(pattern, '', text)

            # Felesleges szóközök és perjelek eltávolítása
            text = re.sub(r'\s+/|/\s+', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()  # Többszörös szóközök és vezető/követő szóközök eltávolítása
            ############################################################

            # Contractions (don't -> do not)
            text = contractions.fix(text)

            # HTML és URL eltávolítása
            text = re.sub(r'<.*?>', '', text)
            text = re.sub(r'http\S+|www\.\S+', '', text)

            # Számok helyettesítése "NUMTOKEN"-nel
            text = re.sub(r'\b\d+\b', 'NUMTOKEN ', text)

            # Mondat szegmentáció
            sentences = sent_tokenize(text)

            # Szó tokenizálás + stopword eltávolítás
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

    # Másolat hogy elkerülje az eredeti módosítását
    df = df.copy()

    # Alkalmazása
    df[new_column] = df[text_column].fillna("").apply(clean_text)

    # Return
    return df


## Harmadik, semmi stopszó sem marad bent

def spacy_preproc(df, text_column, new_column):
    if not hasattr(spacy_preproc, "nlp"):
        spacy_preproc.nlp = spacy.load("en_core_web_sm")
        spacy_preproc.stopwords = spacy_preproc.nlp.Defaults.stop_words

    stopwords = spacy_preproc.stopwords

    def clean_text(text):
        try:
            text = remove_dates(text)
            text = remove_phone_numbers(text)
            text = text.lower()

            phrases_to_remove = ['featured image', 'photo by', 'getty images']
            pattern = r'\b(?:' + '|'.join(map(re.escape, phrases_to_remove)) + r')\b/?'
            text = re.sub(pattern, '', text)
            text = re.sub(r'\s+/|/\s+', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()

            text = contractions.fix(text)
            text = re.sub(r'<.*?>', '', text)
            text = re.sub(r'http\S+|www\.\S+', '', text)
            text = re.sub(r'\b\d+\b', 'NUMTOKEN ', text)

            sentences = sent_tokenize(text)
            tokenized_sentences = []
            for sentence in sentences:
                sentence_clean = sentence.translate(str.maketrans('', '', string.punctuation))
                sentence_clean = re.sub(r'[^a-z0-9\s]', '', sentence_clean).strip()
                tokens = word_tokenize(sentence_clean)
                filtered_tokens = [
                    word for word in tokens
                    if word not in stopwords and word not in {"s", "t"}
                ]
                tokenized_sentences.append(filtered_tokens)

            return tokenized_sentences

        except Exception as e:
            print(f"Error processing text: {text}. Error: {e}")
            return []

    df = df.copy()
    df[new_column] = df[text_column].fillna("").apply(clean_text)
    return df


## Negyedik, a számok szöveggévaló átalakítása ##

def number_preproc(df, text_column, new_column):
    # SpaCy és inflect inicializálás egyszer
    if not hasattr(number_preproc, "nlp"):
        number_preproc.nlp = spacy.load("en_core_web_sm")
        number_preproc.stopwords = number_preproc.nlp.Defaults.stop_words

    if not hasattr(number_preproc, "inflect_engine"):
        number_preproc.inflect_engine = inflect.engine()

    stopwords = number_preproc.stopwords
    inflect_engine = number_preproc.inflect_engine

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

            # A kifejezésekből regex minta készítése, figyelembe véve az opcionális perjelet és a szóhatárokat
            pattern = r'\b(?:' + '|'.join(map(re.escape, phrases_to_remove)) + r')\b/?'

            # A mintázat alkalmazása a szövegre
            text = re.sub(pattern, '', text)

            # Felesleges szóközök és perjelek eltávolítása
            text = re.sub(r'\s+/|/\s+', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            ############################################################

            # Contractions (don't -> do not)
            text = contractions.fix(text)

            # HTML és URL eltávolítása
            text = re.sub(r'<.*?>', '', text)
            text = re.sub(r'http\S+|www\.\S+', '', text)

            # Számok helyettesítése angol szavakkal
            text = re.sub(r'\b\d+\b', num_to_words, text)

            # Mondat szegmentáció
            sentences = sent_tokenize(text)

            # Szó tokenizálás + stopword eltávolítás
            tokenized_sentences = []
            for sentence in sentences:
                # Írásjelek eltávolítása
                sentence_clean = sentence.translate(str.maketrans('', '', string.punctuation))
                sentence_clean = re.sub(r'[^a-z0-9\s]', '', sentence_clean).strip()

                # Tokenizálás
                tokens = word_tokenize(sentence_clean)

                # Stopword szűrés (minden spacy stopwordöt eltávolítunk)
                filtered_tokens = [word for word in tokens if word not in stopwords]

                tokenized_sentences.append(filtered_tokens)

            return tokenized_sentences  # Listában tokenizált mondatok

        except Exception as e:
            print(f"Error processing text: {text}. Error: {e}")
            return []

    df = df.copy()
    df[new_column] = df[text_column].fillna("").apply(clean_text)
    return df



#################### Masodik lepes nyelvi dolgok (lemmatizalas, stemming), 2 kulonbozo modszer ehhez ####################

## Első a lemmatizálós ##

def lemmat_processing(df, text_column, new_column):
    # Inicializálás csak egyszer, ha még nem történt meg
    if not hasattr(lemmat_processing, "lemmatizer"):
        lemmat_processing.lemmatizer = WordNetLemmatizer()

        def get_wordnet_pos(tag):
            if tag.startswith('J'):
                return wordnet.ADJ
            elif tag.startswith('V'):
                return wordnet.VERB
            elif tag.startswith('N'):
                return wordnet.NOUN
            elif tag.startswith('R'):
                return wordnet.ADV
            else:
                return wordnet.NOUN

        lemmat_processing.get_wordnet_pos = get_wordnet_pos

    lemmatizer = lemmat_processing.lemmatizer
    get_wordnet_pos = lemmat_processing.get_wordnet_pos

    def process_text(tokenized_sentences):
        if isinstance(tokenized_sentences, str):
            try:
                tokenized_sentences = ast.literal_eval(tokenized_sentences)
            except Exception as e:
                print(f"Error converting string to list: {e}")
                tokenized_sentences = []

        processed_sentences = []
        for sentence_tokens in tokenized_sentences:
            if not isinstance(sentence_tokens, list):
                continue

            pos_tags = pos_tag(sentence_tokens)
            lemmatized_tokens = []
            for token, tag in pos_tags:
                if token == "numtoken":
                    lemmatized_tokens.append(token)
                else:
                    wordnet_pos = get_wordnet_pos(tag)
                    lemmatized_token = lemmatizer.lemmatize(token, wordnet_pos)
                    lemmatized_tokens.append(lemmatized_token)

            processed_sentences.append(lemmatized_tokens)
        return processed_sentences

    df = df.copy()
    df[new_column] = df[text_column].apply(process_text)

    all_words = [word for sublist in df[new_column] for sentence in sublist for word in sentence if word != "numtoken"]
    word_counts = Counter(all_words)

    df[new_column] = df[new_column].apply(
        lambda sentences: [
            [word for word in sentence if word_counts[word] > 2]
            for sentence in sentences
        ]
    )

    df[new_column] = df[new_column].apply(
        lambda sentences: [
            [word for word in sentence if word.lower() not in ['reuters', 'washington']]
            for sentence in sentences
        ]
    )

    return df


## Masodik a stemming-es ##


def stemming_processing(df, text_column, new_column):
    # Stemmer csak egyszer inicializálódik
    if not hasattr(stemming_processing, "stemmer"):
        stemming_processing.stemmer = PorterStemmer()
    stemmer = stemming_processing.stemmer

    def process_text(tokenized_sentences):
        if isinstance(tokenized_sentences, str):
            try:
                tokenized_sentences = ast.literal_eval(tokenized_sentences)
            except Exception as e:
                print(f"Error converting string to list: {e}")
                tokenized_sentences = []

        processed_sentences = []
        for sentence_tokens in tokenized_sentences:
            if not isinstance(sentence_tokens, list):
                continue

            pos_tags = pos_tag(sentence_tokens)
            stemmed_tokens = []
            for token, tag in pos_tags:
                if token == "numtoken":
                    stemmed_tokens.append(token)
                else:
                    stemmed_token = stemmer.stem(token)
                    stemmed_tokens.append(stemmed_token)

            processed_sentences.append(stemmed_tokens)
        return processed_sentences

    df = df.copy()
    df[new_column] = df[text_column].apply(process_text)

    all_words = [word for sublist in df[new_column] for sentence in sublist for word in sentence if word != "numtoken"]
    word_counts = Counter(all_words)

    df[new_column] = df[new_column].apply(
        lambda sentences: [
            [word for word in sentence if word_counts[word] > 2]
            for sentence in sentences
        ]
    )

    df[new_column] = df[new_column].apply(
        lambda sentences: [
            [word for word in sentence if word.lower() not in ['reuters', 'washington']]
            for sentence in sentences
        ]
    )

    return df



#################### Harmadik lépés a modellezéshez való felkészítés ####################

## Meg kell előtte adni:
# Max szókincs és szekvencia hossz
# MAX_VOCAB_SIZE = 25000
# MAX_LENGTH = 700
# EMBEDDING_DIM = 300  # GloVe 300d



# Tokenizálás és vektorizálás
def prepare_for_modeling_with_glove(tokenized_texts, glove_file,
                                    tokenizer=None, fit_tokenizer=False,
                                    max_vocab_size=25000, max_length=700, embedding_dim=300):
    # Tokenizer setup
    if fit_tokenizer:
        tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
        tokenizer.fit_on_texts(tokenized_texts)

        # "numtoken" biztosítása a vocab-ban
        if "numtoken" not in tokenizer.word_index:
            tokenizer.word_index["numtoken"] = len(tokenizer.word_index) + 1
            tokenizer.index_word[tokenizer.word_index["numtoken"]] = "numtoken"

    # Tokenizált szövegek számsorrá alakítása
    sequences = tokenizer.texts_to_sequences(tokenized_texts)

    # Pad and truncate
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='pre', truncating='post')

    # GloVe embeddings betöltése
    embeddings_index = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector

    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((max_vocab_size, embedding_dim))

    # Embedding matrix
    for word, i in word_index.items():
        if i < max_vocab_size:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    return padded_sequences, embedding_matrix, tokenizer
