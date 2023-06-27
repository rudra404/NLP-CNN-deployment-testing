import torch
import torchtext
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {'GPU' if str(DEVICE) == 'cuda' else 'CPU'}.")
import pandas as pd
import numpy as np
from torchtext.data.functional import to_map_style_dataset
import torchtext.transforms as T
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import time
from sklearn.metrics import f1_score
from model import CNN2
from tokeniser import SpacyTokenizer

train_data = pd.read_csv('train_data.csv')
valid_data = pd.read_csv('valid_data.csv')

train_data = train_data.values.tolist()
valid_data = valid_data.values.tolist()

train_data = to_map_style_dataset(train_data)
valid_data = to_map_style_dataset(valid_data)

#load in glove vocab
text_vocab = torch.load('vocab_obj.pt')

#transformation that will tokenize each text, convert it to vocabulary token IDs and then to padded tensors.
text_transform = T.Sequential(
    SpacyTokenizer(),  # Tokenize
    T.VocabTransform(text_vocab),  # Conver to vocab IDs
    T.ToTensor(padding_value=text_vocab["<pad>"]),  # Convert to tensor and pad
)
def collate_batch(batch):
    labels, texts = zip(*batch)

    texts = text_transform(list(texts))
    labels = torch.tensor(list(labels), dtype=torch.int64)

    return labels.to(DEVICE), texts.to(DEVICE)

def _get_dataloader(data):
    return DataLoader(data, batch_size=64, shuffle=True, collate_fn=collate_batch)

train_dataloader = _get_dataloader(train_data)
valid_dataloader = _get_dataloader(valid_data)

model = CNN2()
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

model = model.to(DEVICE)
criterion = criterion.to(DEVICE)

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
#Define function for training the model
def train(model, iterator, optimizer, criterion):    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    total_preds =np.array([])
    correct =np.array([])
    for batch in tqdm(iterator, desc="\tTraining"):
        optimizer.zero_grad()
                
        labels, texts = batch  # Note that this has to match the order in collate_batch
        predictions = model(texts).squeeze(1)
        loss = criterion(predictions, labels)
        acc = calculate_accuracy(predictions, labels)
        top_preds, y = results(predictions, labels) 
        loss.backward()
        optimizer.step()
        total_preds = np.append(total_preds, top_preds)
        correct = np.append(correct, y)
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    #calcualte f1 score from predictions and correct labels
    f1 = f1_score(correct, total_preds, average='macro')   
    return epoch_loss / len(iterator), epoch_acc / len(iterator), f1
#Define function for evaluating model on validation data
def evaluate(model, iterator, criterion):
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
     
    return epoch_loss / len(iterator), epoch_acc / len(iterator), f1
#Define function for timing training epochs
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# Train the model

N_EPOCHS = 10
#empty lists to store accuracies and F1 scores for each epoch
valid_f1s = []
valid_accs = []
#set value for f1 score to compare results against
best_valid_f1 = 0
print(f"Using {'GPU' if str(DEVICE) == 'cuda' else 'CPU'} for training.")

for epoch in range(N_EPOCHS):
    print(f'Epoch: {epoch+1:02}')
    start_time = time.time()
    
    train_loss, train_acc, train_f1 = train(model, train_dataloader, optimizer, criterion)
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}|  Train. F1: {train_f1*100:.2f}%')
    
    valid_loss, valid_acc, valid_f1  = evaluate(model, valid_dataloader, criterion)
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}|  Val. F1: {valid_f1*100:.2f}%')
    valid_f1s.append(valid_f1)
    valid_accs.append(valid_acc)
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    #only save model if the validation F1 score is an improvement
    if valid_f1 >  best_valid_f1:
        best_valid_f1 = valid_f1
        print("Model saved, epoch number:", epoch+1)
        torch.save(model.state_dict(), 'NLP_model.pt')
print('Best F1 epoch: F1: ', valid_f1s[np.argmax(valid_f1s)], 'Acc: ', valid_accs[np.argmax(valid_f1s)], 'Epoch number: ', np.argmax(valid_f1s)+1)
print('Best Acc epoch: F1: ', valid_f1s[np.argmax(valid_accs)], 'Acc: ', valid_accs[np.argmax(valid_accs)], 'Epoch number: ', np.argmax(valid_accs)+1)
