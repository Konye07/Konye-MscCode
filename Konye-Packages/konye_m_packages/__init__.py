__version__ = "0.1.2"

from .preproc_functions import (
    analyze_text_column,
    plot_most_common_words,
    plot_most_common_words2,
    detect_language,
    clean_dataset,
    remove_dates,
    remove_phone_numbers,
    filtered_preproc,
    preprocess_text,
    spacy_preproc,
    number_preproc,
    lemmat_processing,
    stemming_processing,
    prepare_for_modeling_with_glove,
)

__all__ = [
    "analyze_text_column",
    "plot_most_common_words",
    "plot_most_common_words2",
    "detect_language",
    "clean_dataset",
    "remove_dates",
    "remove_phone_numbers",
    "filtered_preproc",
    "preprocess_text",
    "spacy_preproc",
    "number_preproc",
    "lemmat_processing",
    "stemming_processing",
    "prepare_for_modeling_with_glove",
]

