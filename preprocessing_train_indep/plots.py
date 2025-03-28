from konye_m_packages import analyze_text_column, plot_most_common_words # type: ignore
import pandas as pd # type: ignore
import re
from nltk.corpus import words # type: ignore


####################################
######## Összesen 6 plot ########## 
####################################

#### Először Két ISOT-plot ####

fake_isot = pd.read_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/Isot_Fake.csv")
real_isot = pd.read_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/Isot_True.csv")

print(fake_isot.head())
print(real_isot.head())

fake_isot['label'] = 0  # Fake
real_isot['label'] = 1  # Real

df_fake01 = fake_isot[['text', 'label']]
df_real01 = real_isot[['text', 'label']]

# Plotok
plot_most_common_words(df_fake01, "ISOT-álhírek")

plot_most_common_words(df_real01, "ISOT-igaz hírek")

#################################
#### Második Két Misinf-plot ####
#################################

fake_misinf = pd.read_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/DataSet_Misinfo_FAKE.csv")
real_misinf = pd.read_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/DataSet_Misinfo_TRUE.csv")


fake_misinf['label'] = 0  # Fake
real_misinf['label'] = 1  # Real

df_fake02 = fake_misinf[['text', 'label']]
df_real02 = real_misinf[['text', 'label']]

# Egyberakni mind
combined_isot = pd.concat([df_fake02, df_real02], ignore_index=True)

analyze_text_column(df_fake02, "df_fake02")
analyze_text_column(df_real02, "df_real02")
analyze_text_column(combined_isot, "combined_isot")

# Plotok
plot_most_common_words(df_fake02, "Misinf-Álhírek")

plot_most_common_words(df_real02, "df_real02")


##################################
#### Fülöp-sziget Angol hírek ####
##################################

english_news = pd.read_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/philip_english_news.csv")

# Átnevezés: 'Content' -> 'text'
english_news.rename(columns={'Content': 'text'}, inplace=True)

# Címke csere: 1 -> 0, 0 -> 1
english_news['label'] = english_news['label'].apply(lambda x: 0 if x == 1 else 1)

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

plot_most_common_words(english_news, "Fülöp-szigeteki angol hírek")


################################
###### BS-detector cikkei ######
################################

# Betöltés: hamis hírek
fake_news = pd.read_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/fake_bs_detect.csv")

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

plot_most_common_words(fake_news, "BS-Detector")