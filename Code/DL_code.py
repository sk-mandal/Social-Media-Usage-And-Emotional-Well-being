import pandas as pd
import numpy as np
import random
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

tf.random.set_seed(RANDOM_STATE)

train_path = "train.csv"
val_path = "val.csv"
test_path = "test.csv"

df_train = pd.read_csv(train_path, engine="python", on_bad_lines="skip")
df_val = pd.read_csv(val_path, engine="python", on_bad_lines="skip")
df_test = pd.read_csv(test_path, engine="python", on_bad_lines="skip")

print("Train shape:", df_train.shape)
print("Validation shape:", df_val.shape)
print("Test shape:", df_test.shape)

TARGET = "Dominant_Emotion"

for df in [df_train, df_val, df_test]:
    if "User_ID" in df.columns:
        df.drop(columns=["User_ID"], inplace=True)

df_train = df_train[df_train[TARGET].notna()]
df_val = df_val[df_val[TARGET].notna()]

numeric_features = [
    "Age",
    "Daily_Usage_Time (minutes)",
    "Posts_Per_Day",
    "Likes_Received_Per_Day",
    "Comments_Received_Per_Day",
    "Messages_Sent_Per_Day"
]

categorical_features = ["Gender", "Platform"]

for df in [df_train, df_val, df_test]:
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

label_encoder = LabelEncoder()

label_encoder.fit(
    pd.concat([df_train[TARGET], df_val[TARGET]])
)

y_train = label_encoder.transform(df_train[TARGET])
y_val = label_encoder.transform(df_val[TARGET])
y_test = label_encoder.transform(df_test[TARGET])

print("Emotion classes:", label_encoder.classes_)

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

X_train = preprocessor.fit_transform(df_train[numeric_features + categorical_features])
X_val = preprocessor.transform(df_val[numeric_features + categorical_features])
X_test = preprocessor.transform(df_test[numeric_features + categorical_features])

X_train = X_train.toarray()
X_val = X_val.toarray()
X_test = X_test.toarray()

print("DL input shape:", X_train.shape)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

model = Sequential([
    Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.4),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(len(label_encoder.classes_), activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=1
)

y_val_pred = np.argmax(model.predict(X_val), axis=1)

print("\nANN Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nClassification Report (ANN with Class Weights):")
print(classification_report(
    y_val,
    y_val_pred,
    target_names=label_encoder.classes_,
    zero_division=0
))
