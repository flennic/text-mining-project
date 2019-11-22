import pandas as pd

train = pd.read_csv("data/processed/train.csv", header=None)
test = pd.read_csv("data/processed/test.csv", header=None) 

data = pd.concat([train, test])
data.columns=["category", "title", "review"]

data.to_csv("data/processed/data.csv")

