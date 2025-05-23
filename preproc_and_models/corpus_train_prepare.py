from konye_m_packages import __all__ # type: ignore
from konye_m_packages import analyze_text_column, clean_dataset, detect_language # type: ignore
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

# Plotok külön kódban
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

print(combined_df['label'].value_counts())

# Lefuttatás a data-ra
combined_df = clean_dataset(combined_df)

print("Clean function lefuttatása után: ")
print("combined_df['label'].value_counts()")

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

# Splitting, train és test adatokra

X_train, X_test, y_train, y_test = train_test_split(
    combined_df['text'], combined_df['label'], test_size=0.2, stratify=combined_df['label'], random_state=42
)

# Vissza dataframe-é
train_df = pd.DataFrame({'text': X_train, 'label': y_train})
test_df = pd.DataFrame({'text': X_test, 'label': y_test})

train_df.to_csv('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/train.csv', index=False)
test_df.to_csv('d:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preprocessing_train_indep/databases/test.csv', index=False)

