import joblib
import pandas as pd
from sklearn.metrics import classification_report, f1_score

# Load best model (from step4, retrain on train+val)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

train_df = pd.read_csv("data/Train_clean.csv")
valid_df = pd.read_csv("data/Valid_clean.csv")
test_df  = pd.read_csv("data/Test_clean.csv")

X_all = pd.concat([train_df["clean_text"], valid_df["clean_text"]])
y_all = pd.concat([train_df["label"], valid_df["label"]])
X_test, y_test = test_df["clean_text"], test_df["label"]

# Train final model (TF-IDF + LR )
final_model = Pipeline([
    ("tfidf", TfidfVectorizer(min_df=3, ngram_range=(1,2), sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
])
final_model.fit(X_all, y_all)

# Evaluate
y_pred = final_model.predict(X_test)
print("=== Test Report ===")
print(classification_report(y_test, y_pred, digits=3))
print("Test F1:", f1_score(y_test, y_pred))

# Save
joblib.dump(final_model, "models/best_sentiment_model.pkl")
print("Model saved to models/best_sentiment_model.pkl")