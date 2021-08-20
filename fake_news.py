import os
from functools import partial
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

class NewsPrediction(torch.nn.Module):
    def __init__(self,vocab_length,embedding_dim,max_seq_len,num_layers,num_hidden):
        super(NewsPrediction,self).__init__()
        self.embedding = nn.Embedding(vocab_length,embedding_dim,max_norm=True)
        self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=num_hidden,num_layers=num_layers,batch_first=True)
        self.fc = nn.Linear(max_seq_len*num_hidden,2)
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = torch.reshape(x, (x.size(0),-1,))
        x = self.fc(x)
        return F.log_softmax(x,dim=-1)

class MyDataset(Dataset):
    def __init__(self, ds, train=False):
        super(MyDataset,self).__init__()
        self.data = ds
        self.text = self.data["text"].values
        if train:
            self.labels = self.data["label"].values
        else:
            self.labels = self.data["id"]

    def __getitem__(self, idx):
        labels = self.labels[idx]
        text = self.text[idx]
        if not isinstance(text,str):
            text = "<unk>"
        return labels, text

    def __len__(self):
        return len(self.data)

def yield_tokens(data_iter,tokenizer):
    for _,text in data_iter:
        yield tokenizer(text)

def text_pipeline(text,vocab,tokenizer):
    return vocab(tokenizer(text))

def calculate_max_pad(ds,text_pipeline):
    max_pad = 0
    for d in ds.data:
        t = text_pipeline(d)
        pad = len(t)
        if pad > max_pad:
            max_pad = pad
    return max_pad

def collate_batch(batch,text_pipeline,max_seq_len,device):
    data = torch.zeros((len(batch),max_seq_len),dtype=torch.int64)
    labels = torch.zeros((len(batch)),dtype=torch.int64)
    for index,data_tuple in enumerate(batch):
        processed_text = torch.tensor(text_pipeline(data_tuple[1]), dtype=torch.int64)
        data[index,:len(processed_text)] = processed_text[:max_seq_len]
        labels[index] = data_tuple[0]
    return labels.to(device), data.to(device)

def train(training_data,num_epoch,embedding_dim,lstm_layers,num_hidden,batch_size,weights,device):
    train_data = pd.read_csv(training_data)
    train_ds = MyDataset(train_data,train=True)
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(yield_tokens(train_ds,tokenizer),specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    pipeline = partial(text_pipeline,vocab=vocab,tokenizer=tokenizer)
    max_seq_len = 2500
    model = NewsPrediction(len(vocab),embedding_dim,max_seq_len=max_seq_len,num_layers=lstm_layers,num_hidden=num_hidden)
    
    model = model.to(device)
    col_fn = partial(collate_batch,text_pipeline=pipeline,max_seq_len=max_seq_len,device=device)
    train_dl = DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=0,collate_fn=col_fn)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters())
    num_train = int(len(train_dl)*0.8)
    num_valid = len(train_dl) - num_train
    training, validation = random_split(train_dl, [num_train,num_valid])
    datasets = {"Training":training.dataset, "Validation":validation.dataset}
    best_val_loss = np.inf
    no_improvement = 0
    for epoch in range(num_epoch):
        for d in datasets:
            if d == "Training":
                model.train()
            else:
                model.eval()
            dataset = datasets[d]
            total_pts = 0
            running_loss, running_acc = 0.0, 0.0
            for i, sample in enumerate(dataset):
                labels, data = sample
                optimizer.zero_grad()
                out = model(data)
                _, pred = torch.max(out, 1)
                num_correct = (pred == labels).sum()
                loss = criterion(out,labels)
                if d == "Training":
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()
                running_acc  += num_correct.data.item()
                total_pts += len(sample[0])

            print("Epoch {}, {} Loss: {}, Accuracy: {}".format(epoch + 1, d, running_loss / i, running_acc / total_pts * 100))
            if d == "Validation":
                val_loss = running_loss / i
                if np.array(val_loss) < best_val_loss:
                    torch.save(model.state_dict(),weights)
                    no_improvement = 0
                else:
                    no_improvement += 1
                if no_improvement == 3:
                    break
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint)
    return model, col_fn

def test(testing_data,model,col_fn,weights=None):
    if weights is not None:
        checkpoint = torch.load(weights)
        model = model.load_state_dict(checkpoint)
    data = pd.read_csv(testing_data)
    test_ds = MyDataset(data)
    test_dl = DataLoader(test_ds,batch_size=batch_size,shuffle=False,num_workers=0,collate_fn=col_fn)
    model.eval()
    test_indices = []
    test_labels = []
    for sample in test_dl:
        ID, data = sample
        out = model(data)
        _, pred = torch.max(out, 1)
        test_indices+=ID.cpu().numpy().tolist()
        test_labels+=pred.cpu().numpy().tolist()
    final_df = pd.DataFrame({"id":test_indices,"label":test_labels})
    final_df.set_index("id",inplace=True)
    output = os.path.dirname(testing_data)
    final_df.to_csv(os.path.join(output,"my_submit.csv"))

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    data_dir = r"E:\Data\news_prediction"
    weights_path = r"E:\Data\weights\news_pred.pt"
    num_epoch = 25
    embedding_dim=32
    lstm_layers=1
    num_hidden=32
    batch_size = 1024
    num_workers=os.cpu_count()-1
    train_data = os.path.join(data_dir,"train.csv")
    test_data = os.path.join(data_dir,"test.csv")
    model, col_fn = train(train_data,25,embedding_dim,lstm_layers,num_hidden,batch_size,weights_path,device)
    test(test_data,model,col_fn)
