MSc Szakdolgozat – LSTM és GRU Neurális Hálók Teljesítményének Összehasonlítása Álhírek Osztályozásában Különböző Előfeldolgozási Stratégiákkal

Ez a GitHub repozitórium az Eötvös Loránd Tudományegyetem Survey Statisztika és Adatanalitika mesterszakán készített szakdolgozathoz tartozó teljes kódot tartalmazza.
Szakdolgozat címe: LSTM és GRU neurális hálózatok teljesítményének összehasonlítása álhírek osztályozásában különböző előfeldolgozási stratégiákkal

Szerző: Könye Máté

Év: 2025

Leírás:

A szakdolgozat célja, hogy különböző szöveg-előfeldolgozási stratégiák hatását vizsgálja a mélytanulási modellek –  LSTM és GRU – teljesítményére az álhírek automatikus osztályozásának kontextusában. 
A vizsgálat során többféle előfeldolgozási megközelítést alkalmaztam, majd ezek hatását külön-külön értékeltem azonos adatbázisokon, különféle tanító-, teszt- és független (independent) adathalmazok felhasználásával.


Tartalom:

A repó az alábbi főbb elemeket tartalmazza:

Saját Python csomag – konye_m_packages

A projekt részeként kifejlesztettem egy saját Python csomagot, amely különféle angol nyelvű szövegelőfeldolgozási eljárásokat tartalmaz:
  normálás, kisbetűsítés
  írásjelek, speciális karakterek, URL-ek, HTML tagek eltávolítása
  SpaCy stopword szűrés
  tokenizálás, lemmatizálás, stemming
  modellezésre való előkészítés GloVe támogatásával

A csomag külön mappában található, újrafelhasználható módon, modulárisan megírva.

Előfeldolgozási stratégiák
Öt különböző előfeldolgozási stratégia került kidolgozásra és implementálásra. Mindegyikhez külön kódfájl tartozik, amelyeket háromféle adathalmazra alkalmaztam:
  Train adathalmaz (tanító)
  Test adathalmaz (teszteléshez)
  Independent adathalmaz (általánosíthatóság tesztelésére, külső és saját korpuszra)

LSTM és GRU modellek

A neurális hálók (LSTM és GRU) implementációja Google Colab környezetben történt:
  Modellarchitektúrák
  Betöltött és előkészített adatok
  Tanítási és validálási metrikák
  Konfúziós mátrixok, F1-score, accuracy
  Különböző plotok
Ezek mind le lettek töltve és mindegyik előfeldolgozási módszernél a model mappában vannak megjelenítve.

Korpuszok:
Az összes adat nagy méretük miatt nem szerepelnek közvetlenül a repóban. Az összes használt adathalmaz elérhetősége külön .txt fájlban található, forrásonként megjelölve.
