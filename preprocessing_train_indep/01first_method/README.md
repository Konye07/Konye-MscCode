Az összes modell hasonlóan épül fel és hasonló folyamaton megy keresztül csak a saját fájlaival. 
    Először is a tiszított fájlt a database mappából behívom
    Másodszor, a saját packageből behívott szakdolgozat alapján megfelelő előfeldolgozási függvényeken megy keresztül.
    Az "alap", 01preprocess a következő módon alakul ("alap_preproc"):
        1. filtered_preproc
        2. lemmat_processing
        3. LSTM/GRU-ra előkészítés aminek outputjai:
            padded_train01
            padded_test01
            padded_independent01
            tokenizer
            embedding_matrix
    Ezeket a lépések, a tanító, a tanítóból leválasztott teszt, és a kombinált független adathalmazon lefut.

A 2025 márciusi, általam gyűjtött cikkeken is ez a folyamat, csak külön .py fájlban: "sajat_elso.py".
