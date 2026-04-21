import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(128, 128, batch_first=True)

    def forward(self, x):
        outputs, _ = self.lstm(x)
        return outputs


class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.fc = nn.Linear(128, vocab_size)

    def forward(self, tgt):
        x = self.embedding(tgt)
        out, _ = self.lstm(x)
        return self.fc(out)


class Summarizer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(vocab_size)

    def forward(self, src, tgt):
        enc_out = self.encoder(src)
        return self.decoder(tgt)
