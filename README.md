# Kapitalac 

Kapitalac je backend sistem za automatsku analizu finansijskih izvještaja crnogorskih kompanija i procjenu rizika poslovanja. Sistem prima PDF finansijski izvještaj, izdvaja ključne bilansne pozicije, računa Altman Z-Score i Altman Z'-Score, formira dodatne finansijske pokazatelje, pokreće hibridni ML model i vraća strukturisan JSON odgovor spreman za web aplikaciju.

## Glavna ideja

Cilj projekta je da se statički finansijski izvještaj pretvori u interaktivan kreditno-analitički alat. Umjesto ručnog prepisivanja podataka iz PDF izvještaja, Kapitalac automatski obrađuje dokument i prikazuje procjenu finansijskog zdravlja kompanije.

Sistem je prilagođen formatu finansijskih izvještaja koji su dostupni za kompanije u Crnoj Gori.

## Funkcionalnosti

- Upload i obrada PDF finansijskog izvještaja
- Automatsko izdvajanje finansijskih podataka iz izvještaja
- Validacija kvaliteta izdvojenih podataka
- Originalni Altman Z-Score
- Altman Z'-Score za privatne kompanije
- Pokazatelji likvidnosti, zaduženosti, profitabilnosti i obrta aktive
- Hibridni ML model sa više kandidata
- Automatski izbor aktivnog modela na osnovu metrika
- Procjena rizika po svim ML kandidatima
- Konsenzus modela
- Objašnjenje faktora koji utiču na procjenu rizika
- Drift detection za poređenje novih izvještaja sa trening distribucijom
- MLOps status i MLflow tracking metapodaci
- JSON odgovor prilagođen Lovable frontend aplikaciji

## Modeli

Kapitalac koristi kombinaciju finansijskih formula i mašinskog učenja:

- Altman Z-Score
- Altman Z'-Score
- Logistic Regression
- Random Forest
- XGBoost

Aktivni ML model bira se automatski prema evaluacionim metrikama. Ostali modeli se ne odbacuju, već se njihove procjene prikazuju radi transparentnosti i poređenja.

