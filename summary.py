def summarize(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)

    emb = get_embeddings(inputs['input_ids'], inputs['attention_mask']).to(device)

    tgt = torch.zeros((1, 32), dtype=torch.long).to(device)

    output = model(emb, tgt)
    pred = torch.argmax(output, dim=-1)

    return tokenizer.decode(pred[0], skip_special_tokens=True)


print(summarize("India launched a new satellite today for communication."))

