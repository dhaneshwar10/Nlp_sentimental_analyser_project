import re
import pandas as pd
import nltk
import contractions
from unidecode import unidecode
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation


nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")

lemmatizer=WordNetLemmatizer()
stop_words=set(stopwords.words('english'))
negations={"no","not","nor","n't"}
stop_words={w for w in stop_words if w not in negations}

def preprocess_text(text: str) -> str:
    text=contractions.fix(str(text))
    text=unidecode(text)
    text=text.lower()
    text=re.sub(r"<[^>]+", " ", text)
    text=re.sub(r"https\S+|www\.\S+", "<url>", text)
    tokens=word_tokenize(text)
    tokens=[ t for t in tokens if t not in stop_words and t not in punctuation]
    return "".join(tokens)

for split in ["Train", "Valid", "Test"]:
    df=pd.read_csv(f"data/{split}.csv")
    df["clean_text"]=df["text"].apply(preprocess_text)
    df.to_csv(f"data/{split}_clean.csv",index=False)
    print(f"{split} cleaned and saved to data/ {split}_clean.csv")

