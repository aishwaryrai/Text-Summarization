#Install Libraries
!pip install transformers pandas scikit-learn rouge-score nltk

!pip install kagglehub

import kagglehub

path = kagglehub.dataset_download("gowrishankarp/newspaper-text-summarization-cnn-dailymail")
print("Dataset path:", path)

#Check Folder
import os

for root, dirs, files in os.walk(path):
    print(root)
    print(files)
    print("-"*40)

dataset_path = path + "/cnn_dailymail"

# Load Dataset
import pandas as pd
train_df = pd.read_csv(dataset_path + "/train.csv")
test_df = pd.read_csv(dataset_path + "/test.csv")

print(train_df.head())

import os

dataset_path = "/root/.cache/kagglehub/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail/versions/2"

print(os.listdir(dataset_path))

import os

dataset_path = "/root/.cache/kagglehub/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail/versions/2"

for root, dirs, files in os.walk(dataset_path):
    print("FOLDER:", root)
    print("FILES:", files)
    print("-"*50)




#Reduce Dataset Size 
train_df = train_df[['article', 'highlights']].dropna()
test_df = test_df[['article', 'highlights']].dropna()

train_df = train_df.sample(300, random_state=42)
test_df = test_df.sample(50, random_state=42)
