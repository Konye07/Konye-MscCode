Az összes modell hasonlóan épül fel és hasonló folyamaton megy keresztül csak a saját fájlaival. 
    Először is a tiszított fájlt a database mappából behívom
    Másodszor, a saját packageből behívott szakdolgozat alapján megfelelő előfeldolgozási függvényeken megy keresztül.
    A következő módon alakul a "preproc2_text.py":
        1. preprocess_text
        2. lemmat_processing
        3. LSTM/GRU-ra előkészítés aminek outputjai:
            padded_train02
            padded_test02
            padded_independent02
            tokenizer
            embedding_matrix
    Ezeket a lépések, a tanító, a tanítóból leválasztott teszt, és a kombinált független adathalmazon lefut.

A 2025 márciusi, általam gyűjtött cikkeken is ez a folyamat, csak külön .py fájlban: "sajat_masodik.py".