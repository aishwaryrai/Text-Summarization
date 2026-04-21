from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(df):
    inputs = tokenizer(
        df['article'].tolist(),
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    targets = tokenizer(
        df['highlights'].tolist(),
        max_length=32,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    return inputs, targets

train_inputs, train_targets = tokenize(train_df)
test_inputs, test_targets = tokenize(test_df)
