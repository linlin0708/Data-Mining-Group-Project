# %% [markdown]
# # **Part 1 (FINANCIAL LOAN DATA SIMULATION AND FEATURE ENGINEERING WITH LLM/SLM)**

# %% [markdown]
# ### **Generating Financial Loan Dataset from ChatGPT**

# %%
# import necessary library for LLM Mistral 7B
!pip install torch torchvision torchaudio transformers accelerate
!pip install -U bitsandbytes

# create hugging face token and setup
import os
# remove the hugging face token in order to push to github sucessfully
os.environ["HUGGINGFACE_TOKEN"]="INSERTHUGGINGFACETOKEN"
from huggingface_hub import login
login(token=os.environ["HUGGINGFACE_TOKEN"])

# load llm mistral 7b model and move model to device
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
token = os.environ["HUGGINGFACE_TOKEN"]
llm_mistral = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
llm_mistral.to(device)

# %%
# prompt to LLM (ChatGPT) for data generation

prompt = """

Role:
You are a data simulation assistant helping to generate a realistic financial loan applicant dataset.

Task:
Generate 1500 raw csv format data of synthetic customer records representing individual loan applicants.
The dataset must be realistic, diverse, and internally consistent, but fully synthetic.

Important Rules (VERY IMPORTANT):
Only generate raw, observable customer and financial attributes.
Values should reflect real-world financial constraints (eg., higher income usually correlates with higher balances).


8 Attributes to generate (exactly these):

CustomerID - Unique identifier (e.g., CUST0001, CUST0002, …)
Occupation - Examples: Student, Engineer, Teacher, Sales Executive, Self-Employed, Manager, Clerk, Freelancer, Retired
Monthly Income (MYR) - Numeric (Range: 1,500 – 25,000) and should correlate reasonably with occupation
Account Balance (MYR) - Numeric May be positive or low positive and should generally correlate with income level
Credit Score - Integer (Range: 300 – 850) and distribution should resemble real-world credit scores (more medium-range than extremes)
Total Loan Applied (MYR) - Numeric (Range: 5,000 – 500,000) and larger loans more common for higher-income customers
Loan Duration (Years) - Integer (Range: 1 – 30)
Loan Purpose Text - Short natural-language sentence (10–20 words)

For Loan Purpose Text:
Examples:
“Applying for a personal loan to consolidate existing debts”
“Seeking financing to expand my small business operations”
“Loan needed to cover education expenses and tuition fees”
Text should reflect realistic financial motivations

Output Format:
Output the dataset in CSV format
Include a header row
One row per customer
Ensure no missing values

"""

# %%
# display generated financial dataset by LLM
import pandas as pd

# manual
df = pd.read_csv("loan_applicants_dataset_latest.csv")
df.head(10)

# %% [markdown]
# ### **Fisrt Feature Engineering on Topic Detection based on Loan Purpose Text by SLM BART-MNLI**

# %%
## Topic Detection by SLM
from transformers import pipeline

# Use SLM BART-MNLI in topic detection on loan
topic_slm = pipeline("zero-shot-classification", model = "facebook/bart-large-mnli")
topics = ["Business Expansion","Home Improvement","Education","Vehicle Purchase","Medical Expenses","Emergency Expenses"]
def get_topic(text):
  if pd.isna(text) or text.strip() == "":
    return "Missing"
  result = topic_slm(text, candidate_labels = topics)
  return result['labels'][0]
df['Topic'] = df['Loan Purpose Text'].apply(get_topic)
print(df[['CustomerID', 'Loan Purpose Text', 'Topic']].head(10))

# display aggregate topics
topic_counts = df['Topic'].value_counts()
print("Topic Counts:")
print(topic_counts)

# %%


# %% [markdown]
# ### **Second Feature Engineering on Risk Category based on Topic + Income + Balance + Credit Score by SLM**
# 

# %%
#Risk Catogory using SLM

import numpy as np
import torch
from transformers import pipeline

#Setup SLM using zero-shot classification
device = 0 if torch.cuda.is_available() else -1
risk_slm = pipeline("zero-shot-classification",model="facebook/bart-large-mnli",device=device)
risk_labels = ["high risk", "low risk"]
hypothesis_template = "This loan application is {}."

#Convert numeric features to bands
income_q1, income_q2 = df["Monthly Income (MYR)"].quantile([0.33, 0.66])
bal_q1, bal_q2       = df["Account Balance (MYR)"].quantile([0.33, 0.66])
cs_q1, cs_q2         = df["Credit Score"].quantile([0.33, 0.66])

def band(v, q1, q2):
    if v < q1:
        return "low"
    if v < q2:
        return "medium"
    return "high"

def build_risk_text(row):
    inc_band = band(row["Monthly Income (MYR)"], income_q1, income_q2)
    bal_band = band(row["Account Balance (MYR)"], bal_q1, bal_q2)
    cs_band  = band(row["Credit Score"], cs_q1, cs_q2)
    return (
        f"Loan topic: {row['Topic']}. "
        f"Income level: {inc_band}. Balance level: {bal_band}. Credit score level: {cs_band}. "
        "High income/balance/credit score suggests low risk; low levels suggest high risk."
    )

texts = df.apply(build_risk_text, axis=1).tolist()

outputs = []
batch_size = 16

for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    outputs.extend(risk_slm(batch,candidate_labels=risk_labels,hypothesis_template=hypothesis_template,truncation=True))

def get_high_risk_prob(out):
    for lab, score in zip(out["labels"], out["scores"]):
        if lab.lower() == "high risk":
            return float(score)
    return float("nan")

risk_high_prob = np.array([get_high_risk_prob(o) for o in outputs], dtype=float)

desired_high_rate = 0.25
t = np.nanquantile(risk_high_prob, 1 - desired_high_rate)
df["Risk Category"] = np.where(risk_high_prob >= t, "High", "Low")

for col in ["Risk_High_Prob_SLM", "Risk Category (0.5)"]:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

print("Chosen threshold t =", round(float(t), 4))
print(df["Risk Category"].value_counts())
print(df["Risk Category"].value_counts(normalize=True).round(3))

# %%
# displaying latest dataset (included topic + risk)

df.head(10)

# %% [markdown]
# ### **Exporting New CSV file if Updated or Added New Feature**

# %%
# save for future use by exporting to new csv (run once if anyone of you letak one)

df.to_csv("loan_applicants_dataset_latest.csv", index=False)

# %% [markdown]
# # **Part 2 (PREDICTIVE MODELLING)** **&** **Part 3 (MODEL EVALUATION)**
# 

# %% [markdown]
# ## **2.1 Based on First Feature (Topic Detection) by Decision Tree & Random Forest**

# %% [markdown]
# ### **Decision Tree Modelling**

# %%
## decision tree

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['Encoded_Topic'] = label_encoder.fit_transform(df['Topic'])

X = df[['Monthly Income (MYR)','Account Balance (MYR)','Total Loan Applied (MYR)']]
y = df['Encoded_Topic']

# %%
# split dataset and use balanced weights across each class
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# compute class weights to handle imbalance dataset due to nature imbalance topic detection
classes = np.unique(y_train)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)
weights_dict = dict(zip(classes, class_weights))
sample_weights = y_train.map(weights_dict)

# %% [markdown]
# ### **Evaluation for Decision Tree Model by Classification Report & PieChart Distribution**

# %%
# use decision tree model
from sklearn.metrics import classification_report

dtc = DecisionTreeClassifier(
    max_depth=7,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state = 42
)
dtc.fit(X_train, y_train, sample_weight = sample_weights)

# Predictions and evaluation by classification report
dtc_pred = dtc.predict(X_test)
from sklearn.metrics import classification_report
print("Decision Tree Results with Class Weights")
print(classification_report(y_test, dtc_pred))

# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

classes = np.arange(6)  # all 6 classes: 0 to 5

# create a DataFrame with predictions
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': dtc_pred
})

# count predictions by class, reindex to include all classes
prediction_counts = results_df['Predicted'].value_counts().reindex(classes, fill_value=0)
actual_counts = results_df['Actual'].value_counts().reindex(classes, fill_value=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Donut chart
colors = plt.cm.Set3(np.arange(len(classes)))
wedges, texts, autotexts = ax1.pie(
    prediction_counts.values,
    labels=[f'Class {i}' for i in classes],
    colors=colors,
    autopct='%1.1f%%',
    pctdistance=0.85,
    startangle=90
)
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
ax1.add_artist(centre_circle)
ax1.set_title('DecisionTree Predictions Distribution\n(Donut Chart)', fontsize=14, fontweight='bold')
ax1.axis('equal')

# Bar chart: Actual vs Predicted
x = np.arange(len(classes))
width = 0.35
ax2.bar(x - width/2, actual_counts.values, width, label='Actual', alpha=0.7)
ax2.bar(x + width/2, prediction_counts.values, width, label='Predicted', alpha=0.7)
ax2.set_xlabel('Class')
ax2.set_ylabel('Count')
ax2.set_title('Comparison: Actual vs Predicted Distribution')
ax2.set_xticks(x)
ax2.set_xticklabels([f'Class {i}' for i in classes])
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()


# %% [markdown]
# ### **Visualise Decision Tree Rule (Only Show Once in This Project)**

# %%
# decision rule visualisation
from sklearn.tree import export_text
`
rules = export_text(model, feature_names=['Monthly Income (MYR)','Account Balance (MYR)','Total Loan Applied (MYR)'])
print("Decision Tree Rules:\n", rules)

# %% [markdown]
# ### **Bias and Under/Overfitting Occured in Decision Tree: Overcome by Apply Bagging Method by Random Forest**

# %% [markdown]
# ### **Random Forest Modelling & Evaluation by Classification Report & PieChart Distribution**

# %%
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

rf.fit(X_train, y_train, sample_weight=sample_weights)

rf_pred = rf.predict(X_test)
print("Random Forest Results")
print(classification_report(y_test, rf_pred))

# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

classes = np.arange(6)

# create a DataFrame with predictions
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': rf_pred
})

# count predictions by class, reindex to include all classes
prediction_counts = results_df['Predicted'].value_counts().reindex(classes, fill_value=0)
actual_counts = results_df['Actual'].value_counts().reindex(classes, fill_value=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Donut chart
colors = plt.cm.Set3(np.arange(len(classes)))
wedges, texts, autotexts = ax1.pie(
    prediction_counts.values,
    labels=[f'Class {i}' for i in classes],
    colors=colors,
    autopct='%1.1f%%',
    pctdistance=0.85,
    startangle=90
)
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
ax1.add_artist(centre_circle)
ax1.set_title('RandomForest Predictions Distribution\n(Donut Chart)', fontsize=14, fontweight='bold')
ax1.axis('equal')

# Bar chart: Actual vs Predicted
x = np.arange(len(classes))
width = 0.35
ax2.bar(x - width/2, actual_counts.values, width, label='Actual', alpha=0.7)
ax2.bar(x + width/2, prediction_counts.values, width, label='Predicted', alpha=0.7)
ax2.set_xlabel('Class')
ax2.set_ylabel('Count')
ax2.set_title('Comparison: Actual vs Predicted Distribution')
ax2.set_xticks(x)
ax2.set_xticklabels([f'Class {i}' for i in classes])
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# %% [markdown]
# # 2.2 Based on second features(Categorical:Risk Category)

# %% [markdown]
# ### **Decision Tree Modelling**

# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Encode Risk Category target
label_encoder_risk = LabelEncoder()
df["Encoded_Risk"] = label_encoder_risk.fit_transform(df["Risk Category"])
high_label = int(label_encoder_risk.transform(["High"])[0])

X = df[["Monthly Income (MYR)","Account Balance (MYR)","Credit Score","Total Loan Applied (MYR)","Loan Duration (Years)"]]
y = df["Encoded_Risk"]
print("Risk label mapping:", dict(zip(label_encoder_risk.classes_, label_encoder_risk.transform(label_encoder_risk.classes_))))
print("X shape:", X.shape, "| y distribution:\n", y.value_counts())

# %% [markdown]
# Split + compute sample weights

# %%
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
weights_dict = dict(zip(classes, class_weights))
sample_weights = y_train.map(weights_dict)

print("Train distribution:\n", y_train.value_counts(normalize=True).round(3))

# %% [markdown]
# Train Decision Tree + classification report and confusion matrix

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

dtc = DecisionTreeClassifier(
    max_depth=7,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42
)

dtc.fit(X_train, y_train, sample_weight=sample_weights)
dtc_pred = dtc.predict(X_test)
print("Decision Tree Results (Risk Category)")
print(classification_report(y_test, dtc_pred, target_names=label_encoder_risk.classes_))

ConfusionMatrixDisplay.from_predictions(y_test, dtc_pred,display_labels=label_encoder_risk.classes_)
plt.title("Decision Tree - Confusion Matrix (Risk Category)")
plt.show()

# %%
from sklearn.metrics import roc_auc_score

proba = dtc.predict_proba(X_test)
high_col_idx = list(dtc.classes_).index(high_label)

y_true_high = (y_test == high_label).astype(int)
y_score_high = proba[:, high_col_idx]

auc = roc_auc_score(y_true_high, y_score_high)
print("Decision Tree ROC AUC (High as positive):", round(auc, 4))

from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_estimator(dtc, X_test, y_test, pos_label=high_label, name="Decision Tree")
plt.title("Decision Tree - ROC Curve (Risk Category)")
plt.show()

# %% [markdown]
# Pie chart distribution

# %%
results_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": dtc_pred
})

pred_counts = results_df["Predicted"].value_counts().sort_index()
labels = [label_encoder_risk.inverse_transform([i])[0] for i in pred_counts.index]

plt.figure(figsize=(5,5))
plt.pie(pred_counts.values, labels=labels, autopct="%1.1f%%")
plt.title("Decision Tree - Predicted Risk Category Distribution")
plt.show()

# %%
#Outcome distribution plots
class_labels = list(label_encoder_risk.classes_)
class_ids = label_encoder_risk.transform(class_labels)
actual_counts = pd.Series(y_test).value_counts().reindex(class_ids, fill_value=0)
pred_counts   = pd.Series(dtc_pred).value_counts().reindex(class_ids, fill_value=0)
fig = plt.figure(figsize=(12, 4))

# Donut plot
ax1 = fig.add_subplot(1, 2, 1)
ax1.pie(
    pred_counts.values,
    labels=class_labels,
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops=dict(width=0.45)
)
ax1.set_title("Decision Tree Predictions Distribution\n(Donut Chart)")

# Actual vs Predicted bar
ax2 = fig.add_subplot(1, 2, 2)
x = np.arange(len(class_labels))
width = 0.35
ax2.bar(x - width/2, actual_counts.values, width, label="Actual")
ax2.bar(x + width/2, pred_counts.values, width, label="Predicted")
ax2.set_xticks(x)
ax2.set_xticklabels(class_labels)
ax2.set_ylabel("Count")
ax2.set_title("Comparison: Actual vs Predicted Distribution")
ax2.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ### **Logistic Regression Modelling**

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

lr = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

lr.fit(X_train, y_train, sample_weight=sample_weights)
lr_pred = lr.predict(X_test)

print("Logistic Regression Results (Risk Category)")
print(classification_report(y_test, lr_pred, target_names=label_encoder_risk.classes_))

ConfusionMatrixDisplay.from_predictions(y_test, lr_pred,display_labels=label_encoder_risk.classes_)
plt.title("Logistic Regression - Confusion Matrix (Risk Category)")
plt.show()

# %%
from sklearn.metrics import roc_auc_score

lr_proba = lr.predict_proba(X_test)
high_col_idx = list(lr.classes_).index(high_label)

y_true_high = (y_test == high_label).astype(int)
y_score_high = lr_proba[:, high_col_idx]

lr_auc = roc_auc_score(y_true_high, y_score_high)
print("Logistic Regression ROC AUC (High as positive):", round(lr_auc, 4))

from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_estimator(lr, X_test, y_test, pos_label=high_label, name="Logistic Regression")
plt.title("Logistic Regression - ROC Curve (Risk Category)")
plt.show()

# %%
pred_counts = pd.Series(lr_pred).value_counts().sort_index()
labels = label_encoder_risk.inverse_transform(pred_counts.index)

plt.figure(figsize=(5,5))
plt.pie(pred_counts.values, labels=labels, autopct="%1.1f%%", startangle=90)
plt.title("Logistic Regression - Predicted Risk Category Distribution")
plt.show()

# %%
#Outcome distribution plots
class_labels = list(label_encoder_risk.classes_)
class_ids = label_encoder_risk.transform(class_labels)
actual_counts = pd.Series(y_test).value_counts().reindex(class_ids, fill_value=0)
pred_counts   = pd.Series(lr_pred).value_counts().reindex(class_ids, fill_value=0)
fig = plt.figure(figsize=(12, 4))

# Donut plot
ax1 = fig.add_subplot(1, 2, 1)
ax1.pie(
    pred_counts.values,
    labels=class_labels,
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops=dict(width=0.45)
)
ax1.set_title("Logistic Regression Predictions Distribution\n(Donut Chart)")

# Actual vs Predicted bar
ax2 = fig.add_subplot(1, 2, 2)
x = np.arange(len(class_labels))
width = 0.35
ax2.bar(x - width/2, actual_counts.values, width, label="Actual")
ax2.bar(x + width/2, pred_counts.values, width, label="Predicted")
ax2.set_xticks(x)
ax2.set_xticklabels(class_labels)
ax2.set_ylabel("Count")
ax2.set_title("Comparison: Actual vs Predicted Distribution")
ax2.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Naive Bayes Modeling

# %%
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay

# train nb
nb = GaussianNB(var_smoothing=1e-9)
nb.fit(X_train, y_train, sample_weight=sample_weights)

# eval
nb_pred = nb.predict(X_test)
print("Naive Bayes Results (Risk Category)")
print(classification_report(y_test, nb_pred, target_names=label_encoder_risk.classes_))

# calculate and print roc auc
nb_proba = nb.predict_proba(X_test)
high_col_idx = list(nb.classes_).index(high_label)
y_true_high = (y_test == high_label).astype(int)
y_score_high = nb_proba[:, high_col_idx]
nb_auc = roc_auc_score(y_true_high, y_score_high)
print(f"\nNaive Bayes ROC AUC (High as positive): {round(nb_auc, 4)}")

# plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay.from_predictions(y_test, nb_pred, display_labels=label_encoder_risk.classes_, ax=ax1)
ax1.set_title("Naive Bayes - Confusion Matrix")

RocCurveDisplay.from_estimator(nb, X_test, y_test, pos_label=high_label, name="Naive Bayes", ax=ax2)
ax2.set_title("Naive Bayes - ROC Curve")

plt.tight_layout()
plt.show()

# %%
#Outcome distribution plots
class_labels = list(label_encoder_risk.classes_)
class_ids = label_encoder_risk.transform(class_labels)
actual_counts = pd.Series(y_test).value_counts().reindex(class_ids, fill_value=0)
pred_counts   = pd.Series(nb_pred).value_counts().reindex(class_ids, fill_value=0)
fig = plt.figure(figsize=(12, 4))

# Donut plot
ax1 = fig.add_subplot(1, 2, 1)
ax1.pie(
    pred_counts.values,
    labels=class_labels,
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops=dict(width=0.45)
)
ax1.set_title("Naive Bayes Predictions Distribution\n(Donut Chart)")

# Actual vs Predicted bar
ax2 = fig.add_subplot(1, 2, 2)
x = np.arange(len(class_labels))
width = 0.35
ax2.bar(x - width/2, actual_counts.values, width, label="Actual")
ax2.bar(x + width/2, pred_counts.values, width, label="Predicted")
ax2.set_xticks(x)
ax2.set_xticklabels(class_labels)
ax2.set_ylabel("Count")
ax2.set_title("Comparison: Actual vs Predicted Distribution")
ax2.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ### XGBoost Modeling

# %%
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score

# Train XGBoost model
xgb = XGBClassifier(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
    random_state=42,
    eval_metric='logloss'
)

xgb.fit(X_train, y_train, sample_weight=sample_weights)

# Evaluation
xgb_pred = xgb.predict(X_test)
print("XGBoost Results (Risk Category)")
print(classification_report(y_test, xgb_pred, target_names=label_encoder_risk.classes_))

# Calculate and print ROC AUC
xgb_proba = xgb.predict_proba(X_test)
high_col_idx = list(xgb.classes_).index(high_label)
y_true_high = (y_test == high_label).astype(int)
y_score_high = xgb_proba[:, high_col_idx]
xgb_auc = roc_auc_score(y_true_high, y_score_high)
print(f"\nXGBoost ROC AUC (High as positive): {round(xgb_auc, 4)}")

# Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay.from_predictions(y_test, xgb_pred, display_labels=label_encoder_risk.classes_, ax=ax1)
ax1.set_title("XGBoost - Confusion Matrix")

RocCurveDisplay.from_estimator(xgb, X_test, y_test, pos_label=high_label, name="XGBoost", ax=ax2)
ax2.set_title("XGBoost - ROC Curve")

plt.tight_layout()
plt.show()

# %%
# Outcome distribution plots
class_labels = list(label_encoder_risk.classes_)
class_ids = label_encoder_risk.transform(class_labels)
actual_counts = pd.Series(y_test).value_counts().reindex(class_ids, fill_value=0)
pred_counts   = pd.Series(xgb_pred).value_counts().reindex(class_ids, fill_value=0)
fig = plt.figure(figsize=(12, 4))

# Donut plot
ax1 = fig.add_subplot(1, 2, 1)
ax1.pie(
    pred_counts.values,
    labels=class_labels,
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops=dict(width=0.45)
)
ax1.set_title("XGBoost Predictions Distribution\n(Donut Chart)")

# Actual vs Predicted bar
ax2 = fig.add_subplot(1, 2, 2)
x = np.arange(len(class_labels))
width = 0.35
ax2.bar(x - width/2, actual_counts.values, width, label="Actual")
ax2.bar(x + width/2, pred_counts.values, width, label="Predicted")
ax2.set_xticks(x)
ax2.set_xticklabels(class_labels)
ax2.set_ylabel("Count")
ax2.set_title("Comparison: Actual vs Predicted Distribution")
ax2.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## **Section 6.0: Model Interpretation & Business Insights (LLM Generated)**

# %%
import numpy as np

# 1. Extract statistical summaries for all models
feature_names = ["Monthly Income", "Account Balance", "Credit Score", "Total Loan Applied", "Loan Duration"]

# DT Importance (Baseline)
dt_importances = dtc.feature_importances_
dt_top_3_idx = np.argsort(dt_importances)[-3:][::-1]
dt_top_3 = [feature_names[i] for i in dt_top_3_idx]

# XGBoost Importance (Best Performer)
xgb_importances = xgb.feature_importances_
xgb_top_3_idx = np.argsort(xgb_importances)[-3:][::-1]
xgb_top_3 = [feature_names[i] for i in xgb_top_3_idx]

# LR Coefficients (Directionality)
lr_coeffs = lr.coef_[0]
lr_top_3_idx = np.argsort(np.abs(lr_coeffs))[-3:][::-1]
lr_top_3 = [feature_names[i] for i in lr_top_3_idx]

# 2. Construct the Prompt for Mistral (Aligned with 4-mark rubric)
insight_prompt = f"""
<s>[INST] Role: Senior Banking Data Analyst.
Task: Summarize findings, interpret feature importance, and provide business insights based on our modeling results.

Model Performance Summary:
- Decision Tree (Baseline): 91% Accuracy, 0.97 Recall (High Safety).
- Logistic Regression: 81% Accuracy.
- Naive Bayes: 81% Accuracy, 0.97 Recall.
- XGBoost (Best): 92% Accuracy, 0.97 Recall.

Feature Importance Data:
- Top Predictors (Decision Tree & XGBoost): {', '.join(set(dt_top_3 + xgb_top_3))}.
- Key Risk Influencers (Logistic Regression): {', '.join(lr_top_3)}.

Please provide the output in this exact structure:
### 1. Executive Summary of Findings
(Briefly summarize model performance, highlighting XGBoost as the winner).

### 2. Interpretation of Feature Importance
(Explain why variables like {dt_top_3[0]} and {dt_top_3[1]} are the strongest predictors of default risk in our dataset).

### 3. Business Insights & Recommendations
(Provide 3 concrete, data-driven strategies for the bank to reduce default rates while maintaining high loan approval safety).
[/INST]
"""

# 3. Generate with Mistral
inputs = tokenizer(insight_prompt, return_tensors="pt").to(device)
outputs = llm_mistral.generate(**inputs, max_new_tokens=600, temperature=0.7, do_sample=True)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 4. Display Result
print(response.split("[/INST]")[-1].strip())
