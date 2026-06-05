# credit-risk-ml

[![Quality and SonarCloud](https://github.com/ML-PSID/credit-risk-ml/actions/workflows/quality-sonar.yml/badge.svg?branch=main)](https://github.com/ML-PSID/credit-risk-ml/actions/workflows/quality-sonar.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=ML-PSID_credit-risk-ml&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=ML-PSID_credit-risk-ml)

Application Flask (full Python) branchée sur le nettoyage et les variables dérivées déjà définis dans l'analyse existante.

## Run

```bash
cd /Users/sylvain/Desktop/PSID/credit-risk-ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Ensuite ouvrir [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Routes

- `/` : contexte + storytelling
- `/dashboard` : visualisations + KPIs + explication des variables
