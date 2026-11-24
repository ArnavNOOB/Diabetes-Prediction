# =============================
# 📘 1. Import Libraries
# =============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# =============================
# 📗 2. Load Data
# =============================
df = pd.read_csv("diabetes.csv")

# =============================
# 📊 3. Basic Info
# =============================
print("Dataset Info:\n")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary Stats:\n", df.describe())

# =============================
# 🚫 4. Replace Zeros with NaN
# =============================
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

# =============================
# 🧩 5. Handle Missing Values
# =============================
df.fillna(df.median(), inplace=True)

# =============================
# 🧹 6. Remove Duplicates
# =============================
df.drop_duplicates(inplace=True)

# =============================
# 📈 7. Outlier Detection & Removal (IQR Method)
# =============================

def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

numeric_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

df = remove_outliers_iqr(df, numeric_cols)

print("\nAfter Outlier Removal:", df.shape)

# =============================
# 📊 8. Visualization (Optional)
# =============================
plt.figure(figsize=(12, 8))
df[numeric_cols].boxplot()
plt.title("Boxplot After Outlier Removal")
plt.xticks(rotation=45)
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# =============================
# ⚖️ 9. Feature Scaling
# =============================
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# =============================
# 🎯 10. Feature-Target Split
# =============================
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# =============================
# 🧪 11. Train-Test Split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================
# ✅ 12. Final Check
# =============================
print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print("\nProcessed DataFrame Head:\n", df.head())




# =============================
# 📘 1. Import Libraries
# =============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# =============================
# 📗 2. Load Data
# =============================
df = pd.read_csv("diabetes.csv")

# =============================
# 📊 3. Basic Info
# =============================
print("Dataset Info:\n")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary Stats:\n", df.describe())

# =============================
# 🚫 4. Replace Zeros with NaN
# =============================
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

# =============================
# 🧩 5. Handle Missing Values
# =============================
df.fillna(df.median(), inplace=True)

# =============================
# 🧹 6. Remove Duplicates
# =============================
df.drop_duplicates(inplace=True)

# =============================
# 📈 7. Outlier Detection & Removal (IQR Method)
# =============================

def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

numeric_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

df = remove_outliers_iqr(df, numeric_cols)

print("\nAfter Outlier Removal:", df.shape)

# =============================
# 📊 8. Visualization (Optional)
# =============================
plt.figure(figsize=(12, 8))
df[numeric_cols].boxplot()
plt.title("Boxplot After Outlier Removal")
plt.xticks(rotation=45)
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# =============================
# ⚖️ 9. Feature Scaling
# =============================
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# =============================
# 🎯 10. Feature-Target Split
# =============================
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# =============================
# 🧪 11. Train-Test Split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================
# ✅ 12. Final Check
# =============================
print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print("\nProcessed DataFrame Head:\n", df.head())



# ===========================================
# 🎯 MODEL TRAINING, EVALUATION & EXTRA METRICS
# ===========================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, hamming_loss
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -------------------------------------------
# 1️⃣ Feature-Target Split
# -------------------------------------------
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# -------------------------------------------
# 2️⃣ Train-Test Split
# -------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------
# 3️⃣ Logistic Regression Model
# -------------------------------------------
log_model = LogisticRegression(max_iter=1000, solver='liblinear')
log_model.fit(X_train, y_train)

y_pred_lr = log_model.predict(X_test)
y_prob_lr = log_model.predict_proba(X_test)[:, 1]

# -------------------------------------------
# 4️⃣ Random Forest Model
# -------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=6, random_state=42
)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# -------------------------------------------
# 5️⃣ Define a helper function for all metrics
# -------------------------------------------
def calculate_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    specificity = TN / (TN + FP) if (TN + FP) else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    roc_auc = roc_auc_score(y_true, y_prob)
    hamming = hamming_loss(y_true, y_pred)

    return {
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "Specificity": round(specificity, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4),
        "Hamming Loss": round(hamming, 4)
    }

# -------------------------------------------
# 6️⃣ Collect all model metrics
# -------------------------------------------
results = {
    "Logistic Regression": calculate_metrics(y_test, y_pred_lr, y_prob_lr),
    "Random Forest": calculate_metrics(y_test, y_pred_rf, y_prob_rf)
}

results_df = pd.DataFrame(results).T
print("\n📊 Final Model Comparison Table:")
print(results_df)

# Save to CSV for research paper
results_df.to_csv("model_comparison_results.csv", index=True)
print("\n✅ Saved as 'model_comparison_results.csv'")

# -------------------------------------------
# 7️⃣ Visualize Confusion Matrices
# -------------------------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Blues')
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Greens')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()

# -------------------------------------------
# 8️⃣ Feature Importance (Random Forest)
# -------------------------------------------
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[:15], y=importances.index[:15])
plt.title("Top 15 Important Features (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

# ===========================================
# 🎯 MODEL TRAINING AND EVALUATION (ALL MODELS + EXTRA METRICS)
# ===========================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, hamming_loss
)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# -------------------------------------------
# 1️⃣ Feature-Target Split
# -------------------------------------------
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# -------------------------------------------
# 2️⃣ Train-Test Split
# -------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------
# 3️⃣ Define All Models
# -------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear'),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Naive Bayes": GaussianNB(),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)
}

# -------------------------------------------
# 4️⃣ Helper Function for Metrics
# -------------------------------------------
def calculate_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    specificity = TN / (TN + FP) if (TN + FP) else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    roc_auc = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    hamming = hamming_loss(y_true, y_pred)

    return {
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "Specificity": round(specificity, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4) if not np.isnan(roc_auc) else "-",
        "Hamming Loss": round(hamming, 4)
    }

# -------------------------------------------
# 5️⃣ Train, Evaluate, and Collect Results
# -------------------------------------------
results = {}

for name, model in models.items():
    print(f"\n🚀 Training {name} model...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = calculate_metrics(y_test, y_pred, y_prob)
    results[name] = metrics

    # Display summary
    print(f"\n📊 {name} Results")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------------------------
# 6️⃣ Save and Display Summary
# -------------------------------------------
results_df = pd.DataFrame(results).T
print("\n🏁 Final Model Comparison Table:")
print(results_df)

results_df.to_csv("all_model_comparison.csv", index=True)
print("\n✅ Saved as 'all_model_comparison.csv'")

# -------------------------------------------
# 7️⃣ Visualize Model Accuracy Comparison
# -------------------------------------------
plt.figure(figsize=(10, 6))
sns.barplot(x=results_df.index, y=results_df['Accuracy'])
plt.title("Model Accuracy Comparison (All Classifiers)")
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# -------------------------------------------
# 8️⃣ Confusion Matrices
# -------------------------------------------
plt.figure(figsize=(15, 8))
for i, (name, model) in enumerate(models.items()):
    plt.subplot(3, 3, i + 1)
    y_pred = model.predict(X_test)
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='coolwarm')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# ==========================================================
# ✅ Step 5: Full Model Comparison with Extra Metrics
# ==========================================================
from sklearn.metrics import roc_auc_score, hamming_loss

# Combine all models from Step 2 and Step 3B
all_models = models.copy()


# Ensure models are trained
if not all_models:
    raise ValueError("⚠️ No trained models found. Please run training cells first.")

# Helper function to calculate extra metrics
def calculate_metrics(y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    accuracy = accuracy_score(y_true, y_pred)
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    specificity = TN / (TN + FP) if (TN + FP) else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    roc_auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None
    hamming = hamming_loss(y_true, y_pred)

    return {
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "Specificity": round(specificity, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4) if roc_auc is not None else "-",
        "Hamming Loss": round(hamming, 4)
    }

# Evaluate all models
results_summary = []
for name, model in all_models.items():
    print(f"\n🚀 Evaluating {name}...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = calculate_metrics(y_test, y_pred, y_prob)
    results_summary.append({"Model": name, **metrics})

# Convert to DataFrame
summary_df = pd.DataFrame(results_summary)
summary_df = summary_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

# Display summary
print("\n📊 Final Model Performance Summary:")
print(summary_df)

# Save to CSV
summary_df.to_csv("model_comparison_final.csv", index=False)
print("\n✅ Results saved as 'model_comparison_final.csv'")

# ==========================================================
# ✅ Step 6: Visualization
# ==========================================================
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="Accuracy", data=summary_df)
plt.title("Model Accuracy Comparison (Classic + Advanced)")
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# Optional: F1-Score Comparison
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="F1-Score", data=summary_df)
plt.title("Model F1-Score Comparison (Classic + Advanced)")
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()


# ==========================================================
# ✅ Step 5: Create Final Single-Metric Summary for Paper (Extended)
# ==========================================================
from sklearn.metrics import (
    classification_report, accuracy_score, roc_auc_score,
    confusion_matrix, hamming_loss
)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------------------
# 1️⃣ Gather all trained models (auto-safe)
# -------------------------------------------
all_models = {}

# Add models from previous steps if they exist
for var_name in ["models_step2", "models_step3"]:
    if var_name in locals():
        all_models.update(eval(var_name))

# Add Logistic Regression and Random Forest if trained
if "log_model" in locals():
    all_models["Logistic Regression"] = log_model

if "rf_model" in locals():
    all_models["Random Forest"] = rf_model

# Stop if no models found
if not all_models:
    raise ValueError("⚠️ No trained models found. Please run training cells first.")

# -------------------------------------------
# 2️⃣ Calculate Weighted Average + Extra Metrics
# -------------------------------------------
summary = []

for name, model in all_models.items():
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Try to compute ROC-AUC if possible
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
        except Exception:
            roc_auc = np.nan

        # Compute confusion matrix and derived metrics
        cm = confusion_matrix(y_test, y_pred)
        TN, FP, FN, TP = cm.ravel()
        specificity = TN / (TN + FP) if (TN + FP) else 0
        hamming = hamming_loss(y_test, y_pred)

        # Append final results
        summary.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(report["weighted avg"]["precision"], 4),
            "Recall": round(report["weighted avg"]["recall"], 4),
            "F1-Score": round(report["weighted avg"]["f1-score"], 4),
            "Specificity": round(specificity, 4),
            "ROC-AUC": round(roc_auc, 4) if not np.isnan(roc_auc) else "-",
            "Hamming Loss": round(hamming, 4)
        })
    except Exception as e:
        print(f"⚠️ Skipping {name}: {e}")

# -------------------------------------------
# 3️⃣ Create & Save Final DataFrame
# -------------------------------------------
summary_df = (
    pd.DataFrame(summary)
    .sort_values(by="Accuracy", ascending=False)
    .reset_index(drop=True)
)

print("\n📊 Final Model Results (Weighted Average + Research Metrics):")
print(summary_df)

# Save as CSV (for research paper)
summary_df.to_csv("final_model_results.csv", index=False)
print("\n✅ Saved as 'final_model_results.csv' — ready for inclusion in your research paper.")

# -------------------------------------------
# 4️⃣ Visualization
# -------------------------------------------
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="Accuracy", data=summary_df, palette="coolwarm")
plt.title("Model Accuracy Comparison (Weighted Avg)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="F1-Score", data=summary_df, palette="viridis")
plt.title("Model F1-Score Comparison (Weighted Avg)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
