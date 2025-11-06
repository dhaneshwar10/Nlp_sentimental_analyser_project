import pandas as pd

train_df=pd.read_csv("Data/Train.csv")
valid_df=pd.read_csv("Data/Valid.csv")
test_df=pd.read_csv("Data/Test.csv")

print("Train data shape: ",train_df.shape)
print("Valid data shape: ",valid_df.shape)
print("Test data shape: ",test_df.shape)

print("\ncolumns",train_df.columns.to_list())
print(train_df.head())

print("\n missing values in the train data")
print(train_df.isnull().sum())

print("\n class balance in the train data")
print(train_df["label"].value_counts(normalize=True).mul(100).round(2))

print("\n duplicates rows in the train data",train_df.duplicated().sum() )


