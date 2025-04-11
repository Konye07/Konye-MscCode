import pandas as pd
from collections import Counter
import string
import spacy

nlp = spacy.load("en_core_web_sm")

df = pd.read_csv("D:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preproc_and_models/02second_method/felrebesoroltak/indep_felrebesorolt_peldak_aggregalt_gru.csv")
df.columns = ["error", "real_label", "predicted_label", "text", "prob"]

false_negative = df[df["real_label"] == 1]
false_positive = df[df["real_label"] == 0]

teljes_teszt = pd.read_csv("D:/Egyetem/01Ma_Survey/Szakdolgozat/kod/Konye-MscCode/preproc_and_models/02second_method/indep_masodik.csv")

def clean_tokens(token_str):
    # Szövegből idézőjelek eltávolítása és split szóköz alapján
    tokens = token_str.replace("'", "").split()
    return [
        token.lower() for token in tokens
        if token.lower() not in nlp.Defaults.stop_words
        and token not in string.punctuation
        and len(token) > 2
    ]

all_teszt_tokens = []
for text in teljes_teszt["batch2"].dropna():
    all_teszt_tokens.extend(clean_tokens(text))
all_teszt_top30 = set([word for word, _ in Counter(all_teszt_tokens).most_common(50)])

def get_filtered_top_words(df, exclude_words):
    all_tokens = []
    for text in df["text"].dropna():
        all_tokens.extend(clean_tokens(text))
    filtered_tokens = [token for token in all_tokens if token not in exclude_words]
    freq = Counter(filtered_tokens)
    return pd.DataFrame(freq.most_common(40), columns=["word", "count"])

fn_top_words = get_filtered_top_words(false_negative, exclude_words=all_teszt_top30)
fp_top_words = get_filtered_top_words(false_positive, exclude_words=all_teszt_top30)

print("False Negative Top 30 szó (szűrve):")
print(fn_top_words)

print("\nFalse Positive Top 30 szó (szűrve):")
print(fp_top_words)
