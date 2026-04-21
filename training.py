model = Summarizer(vocab_size=30522).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=0)

train_embeddings = train_embeddings.to(device)

for epoch in range(3):
    for i in range(0, len(train_df), 16):

        emb = train_embeddings[i:i+16]
        labels = train_targets['input_ids'][i:i+16].to(device)

        decoder_input = labels[:, :-1]
        decoder_target = labels[:, 1:]

        output = model(emb, decoder_input)

        loss = criterion(
            output.reshape(-1, 30522),
            decoder_target.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch:", epoch, "Loss:", loss.item())


