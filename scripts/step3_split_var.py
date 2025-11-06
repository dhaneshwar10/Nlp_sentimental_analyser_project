import pandas as pd

train_df=pd.read_csv("data/Train_clean.csv")
valid_df=pd.read_csv("data/Valid_clean.csv")
test_df=pd.read_csv("data/Test_clean.csv")

x_train, y_train=train_df["clean_text"], train_df["label"]
x_val, y_val    =valid_df["clean_text"], valid_df["label"]
x_test, y_test=test_df["clean_text"], test_df["label"]

print("Trian size:", len(x_train))
print("Validation size:", len(x_val))
print("Test size:", len(x_test))

