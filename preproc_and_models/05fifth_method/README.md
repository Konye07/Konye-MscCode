Az összes modell hasonlóan épül fel és hasonló folyamaton megy keresztül csak a saját fájlaival. 
    Először is a tiszított fájlt a database mappából behívom
    Másodszor, a saját packageből behívott szakdolgozat alapján megfelelő előfeldolgozási függvényeken megy keresztül.
    A következő módon alakul a "preproc5_numberwords.py":
        1. number_preproc2 (a function nem volt jól beállítva ezért át kellett írnom kódon belül)
        2. lemmat_processing
        3. LSTM/GRU-ra előkészítés aminek outputjai:
            padded_train05
            padded_test05
            padded_independent05
            tokenizer
            embedding_matrix
    Ezeket a lépések, a tanító, a tanítóból leválasztott teszt, és a kombinált független adathalmazon lefut.

A 2025 márciusi, általam gyűjtött cikkeken is ez a folyamat, csak külön .py fájlban: "sajat_otodik.py".