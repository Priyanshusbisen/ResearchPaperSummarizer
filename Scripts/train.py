import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import random
from model import *

# Assuming the previous class definitions (Encoder, Decoder, Attention, Seq2Seq) are available here

# Define a dataset class
class CustomDataset(Dataset):
    def __init__(self, src_file, trg_file):
        self.src = torch.from_numpy(np.load(src_file)).long()
        self.trg = torch.from_numpy(np.load(trg_file)).long()

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.trg[idx]

# Define the training loop
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for _, (src, trg) in enumerate(iterator):
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        
        # Flatten the output and target for loss calculation
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

# Main block to setup and start training
if __name__ == '__main__':
    # Hyperparameters and other settings
    INPUT_DIM = 1000  # Example vocabulary size for source
    OUTPUT_DIM = 1000  # Example vocabulary size for target
    ENC_EMB_DIM = 256  # Embedding size
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 512  # Hidden dimension size for the encoder
    DEC_HID_DIM = 512  # Hidden dimension size for the decoder
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    LEARNING_RATE = 0.001
    N_EPOCHS = 10
    CLIP = 1

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

    # Create the model
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    model = Seq2Seq(enc, dec, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming index 0 is padding

    # Load data
    train_dataset = CustomDataset('path/to/src.npy', 'path/to/trg.npy')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Train the model
    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, CLIP)
        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}')
