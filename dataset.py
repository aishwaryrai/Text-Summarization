#Install Libraries
!pip install transformers pandas scikit-learn rouge-score nltk

# Load Dataset
import pandas as pd

dataset_path = "/root/.cache/kagglehub/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail/versions/2/cnn_dailymail"

train_df = pd.read_csv(dataset_path + "/train.csv")
test_df = pd.read_csv(dataset_path + "/test.csv")

print(train_df.head())


#Reduce Dataset Size 
train_df = train_df[['article', 'highlights']].dropna()
test_df = test_df[['article', 'highlights']].dropna()

train_df = train_df.sample(300, random_state=42)
test_df = test_df.sample(50, random_state=42)
