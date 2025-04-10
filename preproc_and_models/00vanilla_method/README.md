A csak normalizálásan átesett korpuszok normalizálása és modelleinek lépései:
        1. Normalizálás
        2. LSTM/GRU-ra előkészítés aminek outputjai:
            padded_train00
            padded_test00
            padded_independent00
            tokenizer
            embedding_matrix
    Ezeket a lépések, a tanító, a tanítóból leválasztott teszt, és a kombinált független adathalmazon lefut.

A 2025 márciusi, általam gyűjtött cikkeken is ez a folyamat, csak külön .py fájlban: "sajat_nulladik.py".