
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

train_df = pd.read_csv("data/Train_clean.csv")
valid_df = pd.read_csv("data/Valid_clean.csv")

x_train, y_train = train_df["clean_text"], train_df["label"]
x_val, y_val     = valid_df["clean_text"], valid_df["label"]

bow = CountVectorizer(min_df=3)
tfidf = TfidfVectorizer(min_df=3, ngram_range=(1,2), sublinear_tf=True)

models = {
    "BoW + LogisticRegression": Pipeline([("vec", bow), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
    "BoW + NaiveBayes": Pipeline([("vec", bow), ("clf", MultinomialNB())]),
    "BoW + DecisionTree": Pipeline([("vec", bow), ("clf", DecisionTreeClassifier(max_depth=20))]),
    "TFIDF + LogisticRegression": Pipeline([("vec", tfidf), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
    "TFIDF + NaiveBayes": Pipeline([("vec", tfidf), ("clf", MultinomialNB())]),
    "TFIDF + DecisionTree": Pipeline([("vec", tfidf), ("clf", DecisionTreeClassifier(max_depth=20))]),
}

val_scores = {}
for name, pipe in models.items():
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_val)
    val_scores[name] = {"accuracy": accuracy_score(y_val, y_pred), "f1": f1_score(y_val, y_pred)}
    print(name, val_scores[name])

best_model_name = max(val_scores, key=lambda k: val_scores[k]["f1"])
print("Best model on validation:", best_model_name)