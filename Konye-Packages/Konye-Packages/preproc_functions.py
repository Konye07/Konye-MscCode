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

