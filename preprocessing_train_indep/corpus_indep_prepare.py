from konye_m_packages import __all__
from konye_m_packages import analyze_text_column, clean_dataset
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from langdetect import detect
import re
import nltk
nltk.download('words')
from nltk.corpus import words

#####Először fülöp-szigeteki angol hírek #####
# Betöltés philip_english_news

english_news = pd.read_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/philip_english_news.csv")

# Mennyi sora van
print(f"Eredeti sorok száma: {len(english_news)}")

# Átnevezés: 'Content' -> 'text'
english_news.rename(columns={'Content': 'text'}, inplace=True)

# Címke csere: 1 -> 0, 0 -> 1
english_news['label'] = english_news['label'].apply(lambda x: 0 if x == 1 else 1)

print(analyze_text_column(english_news, "english_news"))

# Nyelvi ellenőrzés
# Angol szavak listájának betöltése
english_vocab = set(words.words())

def filter_english(text):
    # Szavak tokenizálása, nem alfanumerikus karakterek eltávolítása
    words_in_text = re.findall(r'\b[a-zA-Z]+\b', text)

    # Csak az angol szókincsben szereplő szavakat tartjuk meg, kivéve 'sa' és 'na'
    filtered_words = [word for word in words_in_text if word.lower() in english_vocab and word.lower() not in {'sa', 'na'}]

    return ' '.join(filtered_words)

# Szűrt szövegek
english_news['text'] = english_news['text'].apply(filter_english)



#### Másodjára a bs_detect ####

# Betöltés: hamis hírek
fake_news = pd.read_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/fake_bs_detect.csv")

# Eredeti sorok száma
print(f"Eredeti sorok száma: {len(fake_news)}")

# Felesleges oszlopok eltávolítása
columns_to_drop = [
    "uuid", "ord_in_thread", "author", "published", "language", "crawled", "site_url",
    "country", "domain_rank", "thread_title", "spam_score", "main_img_url",
    "replies_count", "participants_count", "likes", "comments", "shares", "type"
]
fake_news.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Címke oszlop hozzáadása (minden érték: 0 mert mind fake)
fake_news['label'] = 0


# Csak szükséges oszlopok megtartása
english_news = english_news[['label', 'text', 'text_length']]

analyze_text_column(fake_news, "fake_news")
fake_news = fake_news[['label', 'text', 'text_length']]


# Egyesítés: angol hírek + hamis hírek
combined_data = pd.concat([english_news, fake_news], ignore_index=True)
print(f"Sorok egyesítés után: {len(combined_data)}")
analyze_text_column(combined_data, "combined_data")

# Duplikációk eltávolítása
combined_data_dedup = combined_data.drop_duplicates()
print(f"Sorok duplikáció eltávolítás után: {len(combined_data_dedup)}")
print(f"Eltávolított duplikációk: {len(combined_data) - len(combined_data_dedup)}")

# Betöltés eredeti dataset
existing_data = pd.read_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/train.csv")
print(f"eredeti dataset dataset sorai: {len(existing_data)}")

# # Ellenőrzini hogy van e ugyanolyan hír mindkettőben, ha igen akkor a fgtln-ből removeolni
df_independent = combined_data_dedup[~combined_data_dedup['text'].isin(existing_data['text'])]
print(f"Sorok végső duplikáció eltávolítás után: {len(df_independent)}")
print(f"További eltávolított duplikációk: {len(combined_data_dedup) - len(df_independent)}")
analyze_text_column(df_independent, "df_independent")

df_independent = clean_dataset(df_independent)

# balanceolni a fake-real arányt
# Elkülöníteni a két set-et
df_majority = df_independent[df_independent['label'] == 1]
df_minority = df_independent[df_independent['label'] == 0]

# Kisebbé tenni a realnewst
df_majority_downsampled = resample(df_majority,
                                   replace=False,
                                   n_samples=len(df_minority),  # Egyezzen meg a fake számával
                                   random_state=42)

balanced_df = pd.concat([df_majority_downsampled, df_minority])

df_independent = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(df_independent['label'].value_counts())


# Mentés: végső deduplikált adatok
df_independent.to_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/prepareed_independent.csv", index=False)