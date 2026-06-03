# Kapitalac

Kapitalac je AI alat za analizu finansijskih izvještaja crnogorskih kompanija. Sistem automatski čita PDF finansijske izvještaje, izvlači ključne podatke iz bilansa, računa Altman Z-Score, Altman Z'-Score, finansijske pokazatelje i ML procjenu rizika poslovanja.

## Funkcionalnosti

- Upload PDF finansijskog izvještaja
- Ekstrakcija podataka iz crnogorskih finansijskih obrazaca
- Altman Z-Score
- Altman Z'-Score za privatne firme
- Finansijski pokazatelji:
  - likvidnost
  - zaduženost
  - profitabilnost
  - obrt aktive
- Validacija kvaliteta ekstrakcije
- ML procjena rizika
- JSON format spreman za Lovable frontend

## Tehnologije

- Python
- FastAPI
- pdfplumber
- pandas
- scikit-learn
- joblib
- Logistic Regression
- Lovable frontend

## Struktura projekta

```text
bankarko/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── pdf_extractor.py
│   ├── altman_score.py
│   ├── validation.py
│   ├── ml_model.py
│   ├── train_model.py
│   ├── build_dataset_from_pdfs.py
│   ├── audit_dataset.py
│   └── prepare_training_dataset.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── kapitalac_ml_model.joblib
│
├── docs/
│   └── lovable_api_contract.md
│
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md