# Smart Ticket Understanding Engine
## TCS Mini Project 1 — Project Report

---

## 1. Project Overview

### 1.1 Problem Statement
In any large enterprise, the IT service desk receives hundreds of support tickets daily via email, chat, and web portals. Manually reading, categorizing, and routing each ticket is:
- **Time-consuming** — analysts spend 5–10 minutes per ticket just classifying it
- **Error-prone** — human fatigue leads to mis-routing and incorrect prioritization
- **Inconsistent** — different analysts may classify the same ticket differently

### 1.2 Our Solution
We built an **AI-powered Smart Ticket Understanding Engine** that automatically reads any service desk ticket and classifies it across **5 dimensions**:

| Dimension | What It Does | Method Used |
|-----------|-------------|-------------|
| **Incident Type** (Category) | Identifies what the issue is about (e.g., Network, Hardware, Software) | TF-IDF + Machine Learning |
| **Priority** | Determines urgency level (Critical / High / Medium / Low) | TF-IDF + Machine Learning |
| **Department** | Routes to the correct team (e.g., Infra Team, Desktop Support) | TF-IDF + Machine Learning |
| **Sentiment** | Detects user emotion (Urgent / Frustrated / Neutral / Positive) | VADER Sentiment Analysis |
| **Recommended Action** | Suggests the next step to take | Rule-Based Engine |

### 1.3 Example
**Input Ticket:**
> "Unable to connect VPN since morning, urgent client call in 20 mins."

**AI Output:**
- 📁 Category: **Network/VPN**
- 🔴 Priority: **High**
- 🏢 Route To: **Infra Team**
- 🚨 Sentiment: **Urgent**
- 🎯 Action: **Priority escalation to Infra Team — Assign senior engineer. Target: 1-hour resolution.**

---

## 2. Architecture & System Design

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   STREAMLIT WEB APP                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Single Ticket│  │  CSV Upload  │  │  Accuracy    │  │
│  │   Analysis   │  │  + Batch     │  │  Dashboard   │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘  │
│         │                 │                              │
│  ┌──────▼─────────────────▼──────────────────────────┐  │
│  │           PREDICTION PIPELINE                      │  │
│  │                                                    │  │
│  │  ┌──────────┐  ┌───────────┐  ┌────────────────┐  │  │
│  │  │  Text    │  │  TF-IDF   │  │  ML Classifiers│  │  │
│  │  │Preprocess│──▶│Vectorizer │──▶│  (3 models)   │  │  │
│  │  └──────────┘  └───────────┘  └────────────────┘  │  │
│  │                                                    │  │
│  │  ┌──────────────┐  ┌──────────────────────────┐   │  │
│  │  │   VADER      │  │  Rule-Based Action       │   │  │
│  │  │  Sentiment   │  │  Recommendation Engine   │   │  │
│  │  └──────────────┘  └──────────────────────────┘   │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow (Step by Step)

```
Step 1: User enters ticket text (or uploads CSV)
    │
Step 2: Text Preprocessing (cleaning, lowering, removing noise)
    │
Step 3: TF-IDF Vectorization (converts text → numerical features)
    │
Step 4: ML Classification (3 separate models predict category, priority, department)
    │
Step 5: VADER Sentiment Analysis (independent sentiment scoring)
    │
Step 6: Rule Engine combines (category + priority + sentiment) → recommended action
    │
Step 7: Results displayed on Streamlit dashboard
```

---

## 3. Technology Stack — Libraries & Why We Used Them

### 3.1 Core ML & NLP Libraries

#### 📦 scikit-learn (v1.3+)
**What it is:** The most widely used machine learning library in Python.

**Why we chose it:**
- **Industry standard** — used by companies like Spotify, Booking.com, JP Morgan
- **No GPU required** — runs on any laptop, making it perfect for this project
- **Complete ML pipeline** — provides everything from text vectorization (TF-IDF) to model training and evaluation in one package
- **Well-documented** — extensive documentation with clear examples

**What we used from scikit-learn:**
| Component | Class Used | Purpose |
|-----------|-----------|---------|
| Text Vectorization | `TfidfVectorizer` | Converts raw ticket text into numerical feature vectors |
| Classifier 1 | `LogisticRegression` | Fast, interpretable linear classifier |
| Classifier 2 | `LinearSVC` | Support Vector Machine — excellent for text classification |
| Classifier 3 | `RandomForestClassifier` | Ensemble of decision trees — handles non-linear patterns |
| Pipeline | `Pipeline` | Chains vectorizer + classifier into a single object |
| Evaluation | `classification_report`, `accuracy_score` | Measures model performance |
| Data Split | `train_test_split` | Splits data into 80% training / 20% testing |

#### 📦 vaderSentiment (v3.3.2)
**What it is:** A rule-based sentiment analysis tool specifically designed for social media and short texts.

**Why we chose it:**
- **No training needed** — works out of the box, perfect for ticket text
- **Handles informal text** — understands slang, punctuation emphasis (e.g., "URGENT!!!"), and emoticons
- **Fast** — analyzes sentiment in microseconds per text
- **Proven accuracy** — published in a peer-reviewed paper (Hutto & Gilbert, 2014)

**How it works:**
VADER gives each text a `compound score` from -1 (most negative) to +1 (most positive). We map this to ticket-relevant labels:
- Score ≥ 0.3 → **Positive**
- Score ≤ -0.5 → **Frustrated**
- Contains urgency keywords → **Urgent**
- Otherwise → **Neutral**

#### 📦 nltk (v3.8+)
**What it is:** Natural Language Toolkit — the foundational NLP library in Python.

**Why we chose it:**
- **Tokenization** — splitting text into meaningful words
- **Stop words** — removing common words (the, is, at) that don't help classification
- **Mature library** — 20+ years of development, extremely stable

### 3.2 Data Handling Libraries

#### 📦 pandas (v2.0+)
**What it is:** The go-to library for data manipulation in Python.

**Why we chose it:**
- **DataFrame structure** — perfect for handling tabular ticket data (rows = tickets, columns = attributes)
- **CSV handling** — reads and writes CSV files effortlessly
- **Data analysis** — makes it easy to compute statistics, filter data, and generate reports

**Where it's used:**
- Loading the training dataset (`pd.read_csv`)
- Processing uploaded CSV files in the Streamlit app
- Building result DataFrames for batch predictions

#### 📦 numpy (v1.24+)
**What it is:** Fundamental numerical computing library for Python.

**Why we chose it:**
- **Required dependency** — scikit-learn and pandas are built on top of NumPy
- **Array operations** — efficient numerical computations for ML model internals
- **Confidence scores** — extracting maximum probability values from model predictions

#### 📦 joblib (v1.3+)
**What it is:** Lightweight library for saving and loading Python objects efficiently.

**Why we chose it:**
- **Model persistence** — saves trained scikit-learn models to disk as `.joblib` files
- **Faster than pickle** — optimized for large NumPy arrays (which ML models contain)
- **Recommended by scikit-learn** — it's the official recommended serialization method

**Where it's used:**
- `joblib.dump(model, path)` — saves trained model after training
- `joblib.load(path)` — loads model when the Streamlit app starts

### 3.3 Web Application Libraries

#### 📦 Streamlit (v1.30+)
**What it is:** An open-source Python framework for building ML/data science web apps.

**Why we chose it:**
- **Fastest way to build ML demos** — creates a full web app with just Python (no HTML/CSS/JS needed)
- **Interactive widgets** — text inputs, file uploaders, buttons, progress bars built-in
- **Auto-reload** — automatically refreshes when code changes
- **Used by teams at Google, Amazon, Uber** for internal ML tools
- **Free deployment** — can be deployed to Streamlit Cloud for free

**Features we built with Streamlit:**
| Feature | Streamlit Component |
|---------|-------------------|
| Text input for tickets | `st.text_area()` |
| Classify button | `st.button()` |
| Results display | `st.metric()` |
| CSV file upload | `st.file_uploader()` |
| Download results | `st.download_button()` |
| Progress bars | `st.progress()` |
| Charts | `st.plotly_chart()` |
| Page navigation | `st.sidebar.radio()` |

#### 📦 Plotly (v5.18+)
**What it is:** Interactive visualization library for creating publication-quality charts.

**Why we chose it:**
- **Interactive charts** — users can hover, zoom, and click on data points
- **Beautiful defaults** — produces professional-looking charts without configuration
- **Native Streamlit integration** — works seamlessly with `st.plotly_chart()`

**Charts we created:**
- Pie chart: Category distribution of classified tickets
- Pie chart: Priority distribution of classified tickets
- Bar chart: Model accuracy comparison across targets

#### 📦 openpyxl (v3.1+)
**What it is:** Library for reading/writing Excel files.

**Why we chose it:**
- **Excel support** — allows the app to handle `.xlsx` file uploads in addition to CSV
- **Enterprise compatibility** — many companies use Excel for ticket data exports

---

## 4. Project Structure — File by File

```
AI_project/
├── app.py                      ← Main Streamlit web application
├── requirements.txt            ← All Python dependencies
├── README.md                   ← Setup instructions
├── PROJECT_REPORT.md           ← This report
│
├── data/
│   ├── generate_dataset.py     ← Synthetic training data generator
│   └── tickets.csv             ← 500 generated training tickets
│
├── models/
│   ├── train.py                ← Model training & evaluation pipeline
│   ├── predictor.py            ← Prediction engine (TicketPredictor class)
│   └── saved/                  ← Saved model files (.joblib)
│       ├── category_model.joblib
│       ├── priority_model.joblib
│       ├── department_model.joblib
│       └── training_meta.json
│
├── utils/
│   ├── preprocessing.py        ← Text cleaning & feature extraction
│   └── actions.py              ← Action recommendation engine
│
└── reports/
    └── accuracy_report.md      ← Auto-generated accuracy metrics
```

---

## 5. Detailed Step-by-Step Explanation

### Step 1: Dataset Creation (`data/generate_dataset.py`)

**Why synthetic data?**
Real enterprise ticket data is confidential. We created a **realistic synthetic dataset** of 500 IT support tickets that mimics real-world patterns.

**How it works:**
1. We defined **8 incident categories** with 25 template tickets each (200 base tickets)
2. Each ticket gets randomly assigned a **priority** (weighted: 10% Critical, 25% High, 40% Medium, 25% Low)
3. Each ticket gets a **sentiment modifier** appended (e.g., "Please help ASAP!" for Urgent)
4. Each ticket gets a **priority modifier** prepended (e.g., "CRITICAL:" for Critical)
5. The final dataset has 500 labeled tickets with columns: `ticket_text`, `category`, `priority`, `department`, `sentiment`, `recommended_action`

**The 8 Categories:**
1. Network/VPN → Infra Team
2. Hardware → Desktop Support
3. Software/Application → Application Support
4. Account/Access → IAM Team
5. Email/Communication → Collaboration Team
6. Database → DBA Team
7. Security → Security Team
8. General IT → Service Desk

### Step 2: Text Preprocessing (`utils/preprocessing.py`)

**Why preprocess text?**
Raw ticket text contains noise (emails, URLs, special characters) that confuse ML models. Cleaning improves accuracy significantly.

**What our `clean_text()` function does:**
```
Input:  "URGENT: Can't connect to VPN!!! Email: john@company.com"
Step 1: Lowercase           → "urgent: can't connect to vpn!!! email: john@company.com"
Step 2: Remove emails       → "urgent: can't connect to vpn!!!"
Step 3: Remove URLs         → (no change)
Step 4: Remove ticket IDs   → (no change)
Step 5: Clean punctuation   → "urgent: can't connect to vpn!"
Step 6: Normalize spaces    → "urgent: can't connect to vpn!"
```

### Step 3: TF-IDF Vectorization

**What is TF-IDF?**
TF-IDF (Term Frequency — Inverse Document Frequency) converts text into numerical vectors that ML models can understand.

**How it works (simplified):**
- **TF (Term Frequency):** How often a word appears in a ticket
- **IDF (Inverse Document Frequency):** How rare a word is across all tickets
- **TF-IDF = TF × IDF:** Words that are frequent in one ticket but rare overall get high scores

**Example:**
| Word | TF (in this ticket) | IDF (across all tickets) | TF-IDF Score |
|------|----|----|------|
| "vpn" | 0.15 | 2.3 (rare) | **0.345** (important!) |
| "the" | 0.20 | 0.1 (common) | 0.020 (ignored) |
| "connect" | 0.10 | 1.8 | **0.180** (useful) |

**Our TF-IDF settings:**
- `max_features=5000` — keep top 5000 most informative words
- `ngram_range=(1, 2)` — consider single words AND two-word phrases (e.g., "can't connect")
- `sublinear_tf=True` — use logarithmic TF to prevent long tickets from dominating
- `min_df=2` — ignore words that appear in fewer than 2 tickets
- `max_df=0.95` — ignore words that appear in >95% of tickets (too common)

### Step 4: Model Training (`models/train.py`)

**Training Process:**
1. Load 500 tickets from CSV
2. Clean all ticket text
3. Split into 80% training (400 tickets) and 20% testing (100 tickets)
4. For EACH target (category, priority, department):
   - Train 3 different classifiers
   - Evaluate each on the test set
   - Pick the best one based on accuracy
   - Save the best model to disk

**The 3 Classifiers We Tried:**

| Classifier | How It Works | Strengths |
|-----------|-------------|-----------|
| **Logistic Regression** | Draws linear boundaries between categories. Uses probability-based decision making. | Fast, interpretable, works well with TF-IDF |
| **Linear SVC** (Support Vector Machine) | Finds the optimal hyperplane that maximally separates categories. | Excellent for high-dimensional text data |
| **Random Forest** | Builds 200 decision trees and takes a majority vote. | Handles non-linear patterns, resistant to overfitting |

**Why multiple classifiers?**
Different algorithms perform differently on different types of data. By training all three and comparing, we ensure we pick the best one. This is called **model selection**.

**Key training parameters:**
- `class_weight='balanced'` — handles imbalanced classes (e.g., fewer Critical tickets)
- `random_state=42` — ensures reproducible results
- `n_jobs=-1` — uses all CPU cores for faster training (Random Forest)

### Step 5: Sentiment Analysis (`models/predictor.py`)

**Why separate from ML models?**
Sentiment is best detected by VADER (a specialized tool) rather than a general classifier, because:
- VADER understands punctuation intensity ("help!" vs "HELP!!!")
- VADER handles negations ("not working" = negative)
- VADER works without training data

**Our sentiment mapping:**
```
VADER compound score ≥ 0.3           → Positive  😊
VADER compound score ≤ -0.5          → Frustrated 😤
Contains urgency keywords            → Urgent    🚨
Otherwise                            → Neutral   😐
```

**Urgency keywords we detect:**
`urgent`, `asap`, `immediately`, `emergency`, `critical`, `blocking`, `can't work`, `unable to work`, `down`, `outage`, `production down`, `system down`

### Step 6: Action Recommendation (`utils/actions.py`)

**How it works:**
This is a **rule-based engine** (not ML). It maps the combination of (Category × Priority × Sentiment) to a specific recommended action.

**Example rules:**
| Category | Priority | Sentiment | Action |
|----------|----------|-----------|--------|
| Network/VPN | Critical | Urgent | 🚨 IMMEDIATE ESCALATION — Page on-call Infra Team engineer |
| Hardware | High | Frustrated | ⚡ Priority assignment — Call user to acknowledge & provide ETA |
| Software | Medium | Neutral | 📋 Queue for Application Support — Standard SLA (8 hours) |
| General IT | Low | Positive | 📝 Queue for Service Desk — Respond within 24 hours |

**SLA Targets:**
| Priority | Response Time |
|----------|--------------|
| Critical | 30 minutes |
| High | 2 hours |
| Medium | 8 hours |
| Low | 24 hours |

### Step 7: Streamlit Web App (`app.py`)

**The app has 4 pages:**

1. **🔍 Single Ticket Analysis**
   - User types or pastes a ticket
   - Clicks "Classify Ticket"
   - Sees all 5 classification results instantly

2. **📂 CSV Bulk Upload**
   - User uploads a CSV file with a `ticket_text` column
   - App classifies ALL tickets in batch
   - Results displayed in a table with download option
   - Pie charts show category and priority distribution

3. **📊 Accuracy Report**
   - Shows model performance metrics
   - Bar chart comparing accuracy across targets
   - Training/test set sizes

4. **ℹ️ About**
   - Project description and tech stack summary

---

## 6. Model Evaluation & Results

### 6.1 Training Configuration
- **Dataset Size:** 500 tickets
- **Train/Test Split:** 80/20 (400 training, 100 testing)
- **Stratified Split:** Ensures proportional representation of all categories in both sets

### 6.2 Evaluation Metrics Used

| Metric | What It Measures |
|--------|-----------------|
| **Accuracy** | % of tickets correctly classified |
| **Precision** | Of all tickets predicted as category X, how many were actually X? |
| **Recall** | Of all actual category X tickets, how many did we correctly identify? |
| **F1-Score** | Harmonic mean of Precision and Recall (balanced measure) |

### 6.3 Why We Used `classification_report`
Scikit-learn's `classification_report` gives us precision, recall, and F1-score **for each class**, not just overall accuracy. This is important because:
- A model might have 90% overall accuracy but completely fail on rare categories
- We need to ensure ALL categories are classified well, not just the common ones

---

## 7. Key Concepts Demonstrated

### 7.1 NLP (Natural Language Processing)
- Text cleaning and normalization
- TF-IDF feature extraction
- Sentiment analysis with VADER

### 7.2 Text Classification
- Multi-class classification (8 categories)
- Multi-output classification (3 separate targets)
- Model comparison and selection

### 7.3 Prompt Engineering (Conceptual)
- The action recommendation engine demonstrates how structured prompts/rules can generate contextual responses
- The ticket templates in the dataset show how different phrasings affect classification

### 7.4 Basic LLM API Readiness
- The `TicketPredictor` class is designed with a clean `predict()` interface that could easily be swapped with an LLM API call
- The modular architecture allows replacing scikit-learn with OpenAI/Llama with minimal changes

### 7.5 Dataset Preparation
- Synthetic data generation with realistic distributions
- Balanced class representation
- Data augmentation through modifier combinations

---

## 8. How to Run the Project

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation & Execution (3 Commands)
```bash
# Step 1: Install all dependencies
pip install -r requirements.txt

# Step 2: Generate dataset + Train models
python models/train.py

# Step 3: Launch the web application
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## 9. Deliverables Checklist

| Deliverable | Status | Description |
|------------|--------|-------------|
| ✅ Ticket Classifier App | Complete | Streamlit app with single-ticket analysis |
| ✅ CSV Upload + Prediction | Complete | Bulk upload page with batch classification and CSV download |
| ✅ Accuracy Report | Complete | Auto-generated report at `reports/accuracy_report.md` + interactive dashboard |

---

## 10. Future Improvements

1. **Use a pre-trained LLM** (GPT, Llama) for even higher accuracy
2. **Add real training data** from enterprise ticketing systems (ServiceNow, Jira)
3. **Multi-language support** for global enterprises
4. **Auto-assignment** — integrate with ticketing APIs to auto-route tickets
5. **Feedback loop** — let analysts correct predictions to improve the model over time

---

*Report prepared for TCS Mini Project 1 evaluation*
*Smart Ticket Understanding Engine v1.0*
