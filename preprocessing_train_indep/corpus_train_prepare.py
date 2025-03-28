from konye_m_packages import __all__ # type: ignore
from konye_m_packages import analyze_text_column, plot_most_common_words # type: ignore
import pandas as pd # type: ignore
from sklearn.utils import resample # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from langdetect import detect # type: ignore


#### ISOT fájljai ####

fake_isot = pd.read_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/Isot_Fake.csv")
real_isot = pd.read_csv("d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/Isot_True.csv")

print(fake_isot.head())
print(real_isot.head())

fake_isot['label'] = 0  # Fake
real_isot['label'] = 1  # Real

df_fake01 = fake_isot[['text', 'label']]
df_real01 = real_isot[['text', 'label']]

# Egyberakni mind
combined_isot = pd.concat([df_fake01, df_real01], ignore_index=True)

analyze_text_column(df_fake01, "df_fake01")
analyze_text_column(df_real01, "df_real01")
analyze_text_column(combined_isot, "combined_isot")

# Plotok
# plot_most_common_words(df_fake01, "df_fake01")

# plot_most_common_words(df_real01, "df_real01")

### Misinf fájljai ###

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
#plot_most_common_words(df_fake02, "df_fake02")

#plot_most_common_words(df_real02, "df_real02")



# A négy külön fájl kombinálása
combined_df = pd.concat([df_fake01, df_real01, df_fake02, df_real02], ignore_index=True)
# Ellenőrizni mennyi sor van bent leíráshoz viszonyítva
print(f"A dataframe-nek {len(combined_df)} sora van.")


# Duplikáltak
duplicate_count = combined_df.duplicated(subset=['text']).sum()
print(f"Duplikáltak száma: {duplicate_count}")

# Dupl eltávolít
combined_df = combined_df.drop_duplicates(subset=['text'], keep='first')

# Ellenőrizni mennyi sor van bent leíráshoz viszonyítva
print(f"A dataframe-nek {len(combined_df)} sora van.")

label_counts = combined_df['label'].value_counts()
print(label_counts)

# balanceolni a fake-real arányt
# Elkülöníteni a két set-et
df_majority = combined_df[combined_df['label'] == 1]
df_minority = combined_df[combined_df['label'] == 0]

# Kisebbé tenni a realnewst
df_majority_downsampled = resample(df_majority,
                                   replace=False,
                                   n_samples=len(df_minority),  # Egyezzen meg a fake számával
                                   random_state=42)

balanced_df = pd.concat([df_majority_downsampled, df_minority])

combined_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(combined_df['label'].value_counts())



# Save the deduplicated dataframe to a new CSV file
combined_df.to_csv('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/combined_data_deduplicated.csv', index=False)
combined_df.head()


# Alap tisztitas elvegzese, angolnyelv-ures-stb
# Nyelv felismerő function
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# Függvény a preprocessinghez
def clean_dataset(dataframe):
    # Adatok betöltése
    df = dataframe

    # Eredeti sorok száma
    original_row_count = len(df)

    # Számomra nem szükséges oszlop dropolása
    df = df.drop('text_length', axis=1)

    # A 'text' oszlop hiányzó értékeinek kitöltése üres karakterlánccal
    df['text'] = df['text'].fillna("")

    # Nyelv észlelése minden sorban és egy új oszlop hozzáadása
    df['language'] = df['text'].apply(detect_language)

    # Az olyan sorok kiszűrése, ahol a 'text' üres vagy nem angol nyelvű
    df = df[(df['text'].str.strip() != "") & (df['language'] == 'en')]

    # A sorok új számának rögzítése
    cleaned_row_count = len(df)

    # Az eltávolított sorok számának kiszámítása
    rows_removed = original_row_count - cleaned_row_count

    # Összegzés
    print(f"Eredeti sorok száma: {original_row_count}")
    print(f"Eltávolított sorok száma: {rows_removed}")
    print(f"Sorok száma tisztítás után: {cleaned_row_count}")

    # Ellenőrzés
    remaining_non_english = df[df['language'] != 'en']
    remaining_empty = df[df['text'].str.strip() == ""]

    if not remaining_non_english.empty or not remaining_empty.empty:
        print("Figyelmeztetés: Néhány nem angol nyelvű vagy üres sor marad a szűrés után.")
        print(f"Nem angol sorok: {len(remaining_non_english)}")
        print(f"Üres sorok: {len(remaining_empty)}")
    else:
        print("Az összes nem angol és üres szöveges sort sikeresen eltávolítotva.")

    return df

# Lefuttatás a data-ra
cleaned_df = clean_dataset(combined_df)

# Splitting, train és test adatokra

X_train, X_test, y_train, y_test = train_test_split(
    cleaned_df['text'], cleaned_df['label'], test_size=0.2, stratify=cleaned_df['label'], random_state=42
)

# Vissza dataframe-é
train_df = pd.DataFrame({'text': X_train, 'label': y_train})
test_df = pd.DataFrame({'text': X_test, 'label': y_test})

train_df.to_csv('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/train.csv', index=False)
test_df.to_csv('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/test.csv', index=False)

