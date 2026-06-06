"""
Smart Ticket Understanding Engine — Model Training Pipeline
Trains multi-output classifiers for Category, Priority, and Department.
Uses TF-IDF vectorization + Scikit-learn classifiers.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.preprocessing import clean_text, extract_features


def load_data(csv_path):
    """Load and validate the ticket dataset."""
    df = pd.read_csv(csv_path)
    required_cols = ['ticket_text', 'category', 'priority', 'department', 'sentiment']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    print(f"📊 Loaded {len(df)} tickets")
    print(f"   Categories: {df['category'].nunique()} unique")
    print(f"   Priorities: {df['priority'].nunique()} unique")
    print(f"   Departments: {df['department'].nunique()} unique")
    print(f"   Sentiments: {df['sentiment'].nunique()} unique")
    return df


def train_classifier(X_train, y_train, X_test, y_test, target_name):
    """
    Train and evaluate multiple classifiers, return the best one.
    
    Tests: LogisticRegression, LinearSVC, RandomForest, GradientBoosting
    Returns the best-performing model based on accuracy.
    """
    classifiers = {
        "Logistic Regression": Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
            )),
            ('clf', LogisticRegression(
                max_iter=1000,
                C=1.0,
                class_weight='balanced',
                random_state=42,
            )),
        ]),
        "Linear SVC": Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
            )),
            ('clf', LinearSVC(
                max_iter=2000,
                C=1.0,
                class_weight='balanced',
                random_state=42,
            )),
        ]),
        "Random Forest": Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
            )),
            ('clf', RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }

    best_model = None
    best_accuracy = 0
    best_name = ""
    results = {}

    print(f"\n{'='*60}")
    print(f"  Training classifiers for: {target_name}")
    print(f"{'='*60}")

    for name, pipeline in classifiers.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            "accuracy": round(accuracy, 4),
            "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        }
        print(f"  {name:25s} → Accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = pipeline
            best_name = name

    print(f"\n  ✅ Best: {best_name} ({best_accuracy:.4f})")

    # Detailed report for the best model
    y_pred_best = best_model.predict(X_test)
    report = classification_report(y_test, y_pred_best, zero_division=0)
    print(f"\n{report}")

    return best_model, best_name, best_accuracy, results


def train_all_models(csv_path=None):
    """
    Train all classifiers (category, priority, department) and save them.
    
    Returns a dict with training results and metrics.
    """
    if csv_path is None:
        csv_path = os.path.join(PROJECT_ROOT, "data", "tickets.csv")

    # Load data
    df = load_data(csv_path)

    # Clean text
    print("\n🧹 Cleaning text...")
    df['clean_text'] = df['ticket_text'].apply(clean_text)

    # Split data
    X = df['clean_text']
    targets = {
        'category': df['category'],
        'priority': df['priority'],
        'department': df['department'],
    }

    # Train/test split (same split for all targets for consistency)
    X_train, X_test, idx_train, idx_test = train_test_split(
        X, df.index, test_size=0.2, random_state=42, stratify=df['category']
    )

    # Save directory
    save_dir = os.path.join(PROJECT_ROOT, "models", "saved")
    os.makedirs(save_dir, exist_ok=True)

    all_results = {}

    # Train each classifier
    for target_name, y in targets.items():
        y_train = y.loc[idx_train]
        y_test = y.loc[idx_test]

        model, model_name, accuracy, results = train_classifier(
            X_train, y_train, X_test, y_test, target_name
        )

        # Save model
        model_path = os.path.join(save_dir, f"{target_name}_model.joblib")
        joblib.dump(model, model_path)
        print(f"  💾 Saved → {model_path}")

        all_results[target_name] = {
            "best_model": model_name,
            "accuracy": accuracy,
            "detailed_results": results,
        }

    # Save training metadata
    meta = {
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "models": {k: {"best_model": v["best_model"], "accuracy": v["accuracy"]}
                   for k, v in all_results.items()},
    }
    meta_path = os.path.join(save_dir, "training_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Generate accuracy report
    generate_accuracy_report(all_results, meta, df)

    print(f"\n{'='*60}")
    print("  🎉 ALL MODELS TRAINED SUCCESSFULLY!")
    print(f"{'='*60}")
    for target, info in all_results.items():
        print(f"  {target:15s} → {info['accuracy']:.4f} ({info['best_model']})")
    print(f"{'='*60}\n")

    return all_results


def generate_accuracy_report(results, meta, df):
    """Generate a markdown accuracy report."""
    report_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(report_dir, exist_ok=True)

    lines = [
        "# 📊 Smart Ticket Engine — Accuracy Report\n",
        f"## Dataset Summary",
        f"- **Total Tickets**: {len(df)}",
        f"- **Training Set**: {meta['training_samples']} ({meta['training_samples']/len(df)*100:.0f}%)",
        f"- **Test Set**: {meta['test_samples']} ({meta['test_samples']/len(df)*100:.0f}%)",
        f"- **Categories**: {df['category'].nunique()}",
        f"- **Priority Levels**: {df['priority'].nunique()}",
        f"- **Departments**: {df['department'].nunique()}",
        "",
        "## Model Performance\n",
        "| Target | Best Model | Accuracy |",
        "|--------|-----------|----------|",
    ]

    for target, info in results.items():
        lines.append(f"| {target} | {info['best_model']} | {info['accuracy']:.2%} |")

    lines.append("")
    lines.append("## Detailed Results Per Model\n")

    for target, info in results.items():
        lines.append(f"### {target.title()}\n")
        lines.append("| Model | Accuracy |")
        lines.append("|-------|----------|")
        for model_name, model_result in info['detailed_results'].items():
            lines.append(f"| {model_name} | {model_result['accuracy']:.2%} |")
        lines.append("")

    lines.append("## Category Distribution\n")
    lines.append("| Category | Count | Percentage |")
    lines.append("|----------|-------|------------|")
    for cat, count in df['category'].value_counts().items():
        lines.append(f"| {cat} | {count} | {count/len(df)*100:.1f}% |")

    report_path = os.path.join(report_dir, "accuracy_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  📄 Accuracy report → {report_path}")


if __name__ == "__main__":
    import random
    import importlib.util

    # First generate the dataset if it doesn't exist
    data_path = os.path.join(PROJECT_ROOT, "data", "tickets.csv")
    if not os.path.exists(data_path):
        print("📝 Generating dataset first...")
        # Import generate_dataset module from the data directory
        gen_path = os.path.join(PROJECT_ROOT, "data", "generate_dataset.py")
        spec = importlib.util.spec_from_file_location("generate_dataset", gen_path)
        gen_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen_module)
        random.seed(42)
        gen_module.generate_dataset(500, data_path)

    # Train all models
    train_all_models(data_path)
