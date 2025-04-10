Az összes modell hasonlóan épül fel és hasonló folyamaton megy keresztül csak a saját fájlaival. 
    Először is a tiszított fájlt a database mappából behívom
    Másodszor, a saját packageből behívott szakdolgozat alapján megfelelő előfeldolgozási függvényeken megy keresztül.
    A következő módon alakul a "preproc3_spacy.py":
        1. spacy_preproc
        2. lemmat_processing
        3. LSTM/GRU-ra előkészítés aminek outputjai:
            padded_train03
            padded_test03
            padded_independent03
            tokenizer
            embedding_matrix
    Ezeket a lépések, a tanító, a tanítóból leválasztott teszt, és a kombinált független adathalmazon lefut.

A 2025 márciusi, általam gyűjtött cikkeken is ez a folyamat, csak külön .py fájlban: "sajat_harmadik.py".