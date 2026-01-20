import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# File paths
train_path = "train.csv"
val_path = "val.csv"
test_path = "test.csv"

# Load datasets
df_train = pd.read_csv(train_path, engine="python", on_bad_lines="skip")
df_val = pd.read_csv(val_path, engine="python", on_bad_lines="skip")
df_test = pd.read_csv(test_path, engine="python", on_bad_lines="skip")

print("Train Shape:", df_train.shape)
print("Validation Shape:", df_val.shape)
print("Test Shape:", df_test.shape)

print("\n--- Train Dataset Preview ---")
print(df_train.head())

print("\n--- Dataset Info (Train) ---")
print(df_train.info())

print("\n--- Column Names ---")
print(df_train.columns.tolist())

TARGET = "Dominant_Emotion"
print("\nTarget Variable:", TARGET)

print("\n--- Target Distribution (Train) ---")
print(df_train[TARGET].value_counts())

sns.set(style="whitegrid")

plt.figure(figsize=(8, 4))
df_train[TARGET].value_counts().plot(kind="bar")
plt.title("Distribution of Dominant Emotion (Train)")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

print("\nTarget distribution:")
print(df_train[TARGET].value_counts())

missing = df_train.isnull().sum().sort_values(ascending=False)
print("\nMissing values (Train):")
print(missing[missing > 0])

numeric_cols = df_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "User_ID"]

for col in numeric_cols:
    plt.figure(figsize=(6, 3))
    sns.histplot(df_train[col], bins=30, kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(data=df_train, x="Platform", hue=TARGET)
plt.title("Emotion Distribution Across Platforms")
plt.xlabel("Platform")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
sns.countplot(data=df_train, x="Gender", hue=TARGET)
plt.title("Emotion Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
corr = df_train[numeric_cols].corr()
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap (Numeric Features)")
plt.show()

numeric_cols = df_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "User_ID"]

print("\nGenerating boxplots for numeric features to inspect outliers...")

for col in numeric_cols:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df_train[col])
    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)
    plt.show()

TARGET = "Dominant_Emotion"

for df in [df_train, df_val, df_test]:
    if "User_ID" in df.columns:
        df.drop(columns=["User_ID"], inplace=True)

numeric_features = [
    "Age",
    "Daily_Usage_Time (minutes)",
    "Posts_Per_Day",
    "Likes_Received_Per_Day",
    "Comments_Received_Per_Day",
    "Messages_Sent_Per_Day"
]

categorical_features = [
    "Gender",
    "Platform"
]

for df in [df_train, df_val, df_test]:
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

X_train = df_train[numeric_features + categorical_features].copy()
y_train = df_train[TARGET].copy()

X_val = df_val[numeric_features + categorical_features].copy()
y_val = df_val[TARGET].copy()

X_test = df_test[numeric_features + categorical_features].copy()
y_test = df_test[TARGET].copy()

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

print("\nPreprocessing completed successfully.")
print("Train shape:", X_train_processed.shape)
print("Validation shape:", X_val_processed.shape)
print("Test shape:", X_test_processed.shape)

# Training set
train_mask = y_train.notna()
X_train_processed = X_train_processed[train_mask.values]
y_train = y_train[train_mask]

# Validation set
val_mask = y_val.notna()
X_val_processed = X_val_processed[val_mask.values]
y_val = y_val[val_mask]

print("After removing missing target labels:")
print("Train samples:", X_train_processed.shape[0])
print("Validation samples:", X_val_processed.shape[0])

baseline_results = []

dummy_clf = DummyClassifier(
    strategy="most_frequent",
    random_state=RANDOM_STATE
)

dummy_clf.fit(X_train_processed, y_train)
y_val_pred_dummy = dummy_clf.predict(X_val_processed)

dummy_acc = accuracy_score(y_val, y_val_pred_dummy)
baseline_results.append(("Dummy Classifier", dummy_acc))

print("\nDummy Classifier Accuracy:", dummy_acc)
print("\nClassification Report (Dummy Classifier):")
print(classification_report(y_val, y_val_pred_dummy))

log_reg = LogisticRegression(
    max_iter=1000,
    multi_class="auto",
    random_state=RANDOM_STATE
)

log_reg.fit(X_train_processed, y_train)
y_val_pred_lr = log_reg.predict(X_val_processed)

lr_acc = accuracy_score(y_val, y_val_pred_lr)
baseline_results.append(("Logistic Regression", lr_acc))

print("\nLogistic Regression Accuracy:", lr_acc)
print("\nClassification Report (Logistic Regression):")
print(classification_report(y_val, y_val_pred_lr))

cm = confusion_matrix(y_val, y_val_pred_lr)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix – Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

advanced_results = []

dt_clf = DecisionTreeClassifier(
    random_state=RANDOM_STATE,
    max_depth=None
)

dt_clf.fit(X_train_processed, y_train)
y_val_pred_dt = dt_clf.predict(X_val_processed)

dt_acc = accuracy_score(y_val, y_val_pred_dt)
advanced_results.append(("Decision Tree", dt_acc))

print("\nDecision Tree Accuracy:", dt_acc)
print("\nClassification Report (Decision Tree):")
print(classification_report(y_val, y_val_pred_dt))

rf_clf = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_clf.fit(X_train_processed, y_train)
y_val_pred_rf = rf_clf.predict(X_val_processed)

rf_acc = accuracy_score(y_val, y_val_pred_rf)
advanced_results.append(("Random Forest", rf_acc))

print("\nRandom Forest Accuracy:", rf_acc)
print("\nClassification Report (Random Forest):")
print(classification_report(y_val, y_val_pred_rf))

gb_clf = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    random_state=RANDOM_STATE
)

gb_clf.fit(X_train_processed, y_train)
y_val_pred_gb = gb_clf.predict(X_val_processed)

gb_acc = accuracy_score(y_val, y_val_pred_gb)
advanced_results.append(("Gradient Boosting", gb_acc))

print("\nGradient Boosting Accuracy:", gb_acc)
print("\nClassification Report (Gradient Boosting):")
print(classification_report(y_val, y_val_pred_gb))

tuned_results = []

rf_param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 3]
}

rf_tuned = RandomForestClassifier(
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_grid = GridSearchCV(
    rf_tuned,
    rf_param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

rf_grid.fit(X_train_processed, y_train)

best_rf = rf_grid.best_estimator_

print("\n----- Tuned Random Forest -----")
print("Best Parameters:", rf_grid.best_params_)

y_val_pred_rf = best_rf.predict(X_val_processed)
rf_tuned_acc = accuracy_score(y_val, y_val_pred_rf)

tuned_results.append(("Random Forest (Tuned)", rf_tuned_acc))

print("Validation Accuracy:", rf_tuned_acc)
print("\nClassification Report (Tuned RF):")
print(classification_report(y_val, y_val_pred_rf, zero_division=0))


gb_param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [2, 3],
    "min_samples_split": [2, 5]
}

gb_tuned = GradientBoostingClassifier(
    random_state=RANDOM_STATE
)

gb_grid = GridSearchCV(
    gb_tuned,
    gb_param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

gb_grid.fit(X_train_processed, y_train)

best_gb = gb_grid.best_estimator_

print("\n----- Tuned Gradient Boosting -----")
print("Best Parameters:", gb_grid.best_params_)

y_val_pred_gb = best_gb.predict(X_val_processed)
gb_tuned_acc = accuracy_score(y_val, y_val_pred_gb)

tuned_results.append(("Gradient Boosting (Tuned)", gb_tuned_acc))

print("Validation Accuracy:", gb_tuned_acc)
print("\nClassification Report (Tuned GB):")
print(classification_report(y_val, y_val_pred_gb, zero_division=0))
