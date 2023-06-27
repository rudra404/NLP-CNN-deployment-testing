import torch.nn as nn
import torch.nn.functional as F
import torch
#load in glove vocab and embeddings
text_vocab = torch.load('vocab_obj.pt')
pretrained_embeddings = torch.load('pretrained_embeddings.pt')
#Defining a CNN with one small hidden layer
class CNN2(nn.Module):
    def __init__(self, pretrained_embeddings=pretrained_embeddings, pad_idx=text_vocab["<pad>"], n_filters=100, filter_sizes=[2, 3, 4, 5], output_dim=14, 
                 dropout=0.5):
        
        super().__init__()
                
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True, padding_idx=pad_idx)
        self.embedding_dim = pretrained_embeddings.shape[1]
        
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels = 1, 
                      out_channels = n_filters, 
                      kernel_size = (fs, self.embedding_dim)) 
            for fs in filter_sizes
        ])
        #Additional hidden layer
        self.fc1 = nn.Linear(len(filter_sizes) * n_filters, 100)
        self.fc2 = nn.Linear(100, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]                
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        #Pass output of hidden layer througgh ReLu activation function
        hidden = F.relu(self.fc1(cat))
        
        return self.fc2(hidden)
    