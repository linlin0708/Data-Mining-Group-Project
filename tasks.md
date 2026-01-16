# Project Checklist: Data Mining Group Assignment

## Phase 1: Data Preparation

- [x] **[Shared]** Simulate Financial Dataset (1500 rows)
- [x] **[Shared]** Clean Data & Handle Missing Values
- [x] **[Shared]** Generate `Topic` column using SLM (BART/Mistral)

## Phase 2: Feature Engineering & Modeling

### Task 1: Topic Classification (Owner: ZY)

- [x] Train Decision Tree Model
- [x] Train Random Forest Model
- [x] Evaluate (Classification Report, Pie Chart)

### Task 2: Risk Category Classification (Owner: [John] & MingHang)

- [x] **Generate Target:** Create `Risk Category` column (High/Low) using LLM or Quantile Rules.
- [x] **Pre-processing:** Encode categorical variables (LabelEncoder/OneHot).
- [x] **Train Model 1:** Decision Tree Classifier.
- [x] **Train Model 2:** Logistic Regression (Extract Coefficients).
- [x] **Train Model 3:** Naive Bayes.
- [x] **Evaluation:** Print Classification Report (Precision/Recall).
- [x] **Evaluation:** Plot Confusion Matrix.
- [x] **Evaluation:** Plot ROC-AUC Curve.
- [x] **Evaluation:** Plot Pie Chart of Predicted Risk Distribution.

### Task 3: Loan Amount Regression (Owner: Wendy & JuenKai)

- [ ] **Generate Target:** Create `Approved Amount` column.
- [ ] **Train Model 1:** Decision Tree Regressor.
- [ ] **Train Model 2:** XGBoost.
- [ ] **Evaluation:** RMSE & MAE.

## Phase 3: Insights & Reporting

- [ ] **[John]** Extract feature importance/coefficients from Task 2 models.
- [ ] **[Shared]** Feed results to LLM to generate "Business Insights".
- [ ] **[Shared]** Compile all notebooks into `Final_Project.ipynb`.
- [ ] **[Shared]** Push individual branches to GitHub.
- [ ] **[Shared]** Create Pull Requests & Merge to Main.
- [ ] **[Shared]** Write Final Report (5-7 Pages).

---

## Notes & Changes

- _Current Status:_ Phase 1 and Phase 2 (Task 1 & 2) complete. Task 3 pending. John to extract feature importance next.
