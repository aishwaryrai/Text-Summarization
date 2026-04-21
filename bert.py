import torch
from transformers import AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert = AutoModel.from_pretrained('prajjwal1/bert-tiny').to(device)

#Get Embeddings 
def get_embeddings(input_ids, attention_mask):
    with torch.no_grad():
        outputs = bert(input_ids=input_ids.to(device),
                       attention_mask=attention_mask.to(device))
    return outputs.last_hidden_state.cpu()

#Precompute Embeddings

all_embeddings = []

for i in range(0, len(train_df), 16):
    input_ids = train_inputs['input_ids'][i:i+16]
    attention_mask = train_inputs['attention_mask'][i:i+16]

    emb = get_embeddings(input_ids, attention_mask)
    all_embeddings.append(emb)

train_embeddings = torch.cat(all_embeddings, dim=0)

print(train_embeddings.shape)

