# 📊 Smart Ticket Engine — Accuracy Report

## Dataset Summary
- **Total Tickets**: 496
- **Training Set**: 396 (80%)
- **Test Set**: 100 (20%)
- **Categories**: 8
- **Priority Levels**: 4
- **Departments**: 8

## Model Performance

| Target | Best Model | Accuracy |
|--------|-----------|----------|
| category | Linear SVC | 91.00% |
| priority | Linear SVC | 99.00% |
| department | Linear SVC | 91.00% |

## Detailed Results Per Model

### Category

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 64.00% |
| Linear SVC | 91.00% |
| Random Forest | 71.00% |

### Priority

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 98.00% |
| Linear SVC | 99.00% |
| Random Forest | 98.00% |

### Department

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 64.00% |
| Linear SVC | 91.00% |
| Random Forest | 72.00% |

## Category Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| Security | 62 | 12.5% |
| Hardware | 62 | 12.5% |
| Account/Access | 62 | 12.5% |
| Network/VPN | 62 | 12.5% |
| Software/Application | 62 | 12.5% |
| Database | 62 | 12.5% |
| Email/Communication | 62 | 12.5% |
| General IT | 62 | 12.5% |