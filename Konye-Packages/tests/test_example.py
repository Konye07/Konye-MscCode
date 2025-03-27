from konye_m_packages import analyze_text_column
import pandas as pd


# Tesztadat
df = pd.DataFrame({
    "text": [
        "This is a test sentence with a phone number: 123-456-7890 and a date: 2023-03-27.",
        "Another test with a URL: https://example.com and HTML: <b>bold</b> text."
    ]
})

print(analyze_text_column(df, "df"))