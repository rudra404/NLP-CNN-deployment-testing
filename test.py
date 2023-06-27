import pandas as pd
import numpy as np
from torchtext.data.functional import to_map_style_dataset
import torch
import torchtext.transforms as T
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn import metrics
import matplotlib.pyplot as plt
from model import CNN2
from tokeniser import SpacyTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {'GPU' if str(DEVICE) == 'cuda' else 'CPU'}.")

test_data = pd.read_csv('test_data.csv')
test_data = test_data.values.tolist()
test_data = to_map_style_dataset(test_data)

#load in glove vocab
text_vocab = torch.load('vocab_obj.pt')
    
def collate_batch(batch):
    labels, texts = zip(*batch)

    texts = text_transform(list(texts))
    labels = torch.tensor(list(labels), dtype=torch.int64)

    return labels.to(DEVICE), texts.to(DEVICE)
#transformation that will tokenize each text, convert it to vocabulary token IDs and then to padded tensors.
text_transform = T.Sequential(
    SpacyTokenizer(),  # Tokenize
    T.VocabTransform(text_vocab),  # Conver to vocab IDs
    T.ToTensor(padding_value=text_vocab["<pad>"]),  # Convert to tensor and pad
)

def _get_dataloader(data):
    return DataLoader(data, batch_size=64, shuffle=True, collate_fn=collate_batch)

test_dataloader = _get_dataloader(test_data)

#Define function for calcualting overall accuracy
def calculate_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    _, pred = torch.max(preds, dim=1)
    correct = (pred == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

#Define function for returning list of predictions and correct class labels
#This is used to calculate F1 score
def results(preds, y):
  _, top_pred = torch.max(preds, dim=1)
  top_pred = torch.Tensor.cpu(top_pred)
  y = torch.Tensor.cpu(y)
  top_pred = torch.Tensor.numpy(top_pred)

  return top_pred, y

#Define function for evaluating model on the test data
def test_evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    total_preds = np.array([])
    correct = np.array([])  
    with torch.no_grad():
        for batch in tqdm(iterator, desc="\tEvaluation"):
            labels, texts = batch  # Note that this has to match the order in collate_batch
            predictions = model(texts).squeeze(1)
            loss = criterion(predictions, labels)
            acc = calculate_accuracy(predictions, labels)
            top_preds, y = results(predictions, labels) 
            total_preds = np.append(total_preds, top_preds)
            correct = np.append(correct, y)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    f1 = f1_score(correct, total_preds, average='macro')

    print(f'Test Acc: {(epoch_acc/len(iterator)) *100:.2f}% | Test F1: {f1*100:.2f}%')

    print("Classification report for classifier:\n%s\n"
      % (metrics.classification_report(correct, total_preds,  target_names = ['amusement', 'anger', 'approval', 'confusion', 'curiosity',
       'desire', 'disapproval', 'fear', 'gratitude', 'joy', 'love',
       'remorse', 'sadness', 'neutral'])))
    
    fig, ax = plt.subplots(figsize=(10,10))
    disp = metrics.ConfusionMatrixDisplay.from_predictions(correct, total_preds, labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13], display_labels = ['amusement', 'anger', 'approval', 'confusion', 'curiosity',
       'desire', 'disapproval', 'fear', 'gratitude', 'joy', 'love',
       'remorse', 'sadness', 'neutral'], xticks_rotation='vertical',normalize='true',values_format='.2f', ax=ax)
    disp.figure_.suptitle("Confusion Matrix")

    #print("Confusion matrix:\n%s" % disp.confusion_matrix)
    print(f'Test Acc: {(epoch_acc/len(iterator)) *100:.2f}% | Test F1: {f1*100:.2f}%')
    return epoch_loss / len(iterator), epoch_acc / len(iterator), f1




# #Defining a CNN with one small hidden layer
# class CNN2(nn.Module):
#     def __init__(self, pretrained_embeddings, n_filters, filter_sizes, output_dim, 
#                  dropout, pad_idx):
        
#         super().__init__()
                
#         self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True, padding_idx=pad_idx)
#         self.embedding_dim = pretrained_embeddings.shape[1]
        
#         self.convs = nn.ModuleList([
#             nn.Conv2d(in_channels = 1, 
#                       out_channels = n_filters, 
#                       kernel_size = (fs, self.embedding_dim)) 
#             for fs in filter_sizes
#         ])
#         #Additional hidden layer
#         self.fc1 = nn.Linear(len(filter_sizes) * n_filters, 100)
#         self.fc2 = nn.Linear(100, output_dim)
        
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, text):
#         embedded = self.embedding(text)
#         embedded = embedded.unsqueeze(1)

#         conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]                
#         pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
#         cat = self.dropout(torch.cat(pooled, dim = 1))
#         #Pass output of hidden layer througgh ReLu activation function
#         hidden = F.relu(self.fc1(cat))
        
#         return self.fc2(hidden)
    
# #Use best filter sizing from experiment 2
# N_FILTERS = 100
# FILTER_SIZES = [2, 3, 4, 5]
# OUTPUT_DIM = 14
# DROPOUT = 0.5
# PAD_IDX = text_vocab["<pad>"]
criterion = nn.CrossEntropyLoss()

#Use CNN2 as it the best performing architecture from experiment 3
model = CNN2()
model.to(DEVICE)
criterion = criterion.to(DEVICE)
#Load pretrained parameters for CNN
model.load_state_dict(torch.load('NLP_model.pt', map_location=torch.device('cpu')))

test_loss, test_acc, test_f1 = test_evaluate(model, test_dataloader, criterion)