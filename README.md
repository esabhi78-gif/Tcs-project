<<<<<<< HEAD
# Tcs-project
a simple project based on ticket classfication
=======
# 🎫 Smart Ticket Understanding Engine

AI-powered IT service desk ticket classifier that analyzes tickets across 5 dimensions:
**Category** · **Priority** · **Department** · **Sentiment** · **Recommended Action**

## Quick Start (3 Commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset + Train models
python models/train.py

# 3. Launch the app
streamlit run app.py
```

## Project Structure

```
AI_project/
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── data/
│   ├── generate_dataset.py   # Synthetic ticket data generator
│   └── tickets.csv           # Training dataset (auto-generated)
├── models/
│   ├── train.py              # Model training pipeline
│   ├── predictor.py          # Prediction engine
│   └── saved/                # Saved model files
├── utils/
│   ├── preprocessing.py      # Text cleaning utilities
│   └── actions.py            # Action recommendation engine
└── reports/
    └── accuracy_report.md    # Auto-generated accuracy report
```

## Features

- **Single Ticket Analysis** — Paste any ticket text and get instant classification
- **CSV Bulk Upload** — Upload a CSV file and classify hundreds of tickets at once
- **Accuracy Dashboard** — View model performance metrics and charts
- **Downloadable Results** — Export predictions as CSV

## Tech Stack

| Library | Purpose |
|---------|---------|
| scikit-learn | ML classification (TF-IDF + LogReg/SVM/RF) |
| vaderSentiment | Sentiment analysis |
| streamlit | Web application |
| plotly | Interactive charts |
| pandas | Data processing |
| nltk | NLP preprocessing |

## Classification Taxonomy

**8 Categories**: Network/VPN, Hardware, Software/Application, Account/Access, Email/Communication, Database, Security, General IT

**4 Priority Levels**: Critical, High, Medium, Low

**8 Departments**: Infra Team, Desktop Support, Application Support, IAM Team, Collaboration Team, DBA Team, Security Team, Service Desk

**4 Sentiments**: Urgent, Frustrated, Neutral, Positive
>>>>>>> fee1257 (Initial commit)
