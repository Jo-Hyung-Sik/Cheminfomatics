from utility.Featurizer import *
from torch_geometric.loader import DataLoader
from torch import nn
from GCN.model import GCN
import pandas as pd

from sklearn.metrics import * 

def training(dataloader, model, optimizer, criterion):
    model.train()
    accuracy = 0
    train_loss = 0
    for (k, data) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        train_loss += loss
        loss.backward()
        
        optimizer.step()
        preds = output.argmax(dim=1)
        # running_accuracy += torch.sum(preds == data.y).detach().cpu().numpy()/data.batch.size(0)
        accuracy += accuracy_score(data.y, preds)
    print('training :', accuracy / len(dataloader))

    return model, train_loss

def eval(dataloader, model, criterion):
    model.eval()
    accuracy = 0
    eval_loss = 0
    for (k, data) in enumerate(dataloader):
        output = model(data)
        preds = output.argmax(dim=1)
        accuracy += accuracy_score(data.y, preds)
        loss = criterion(output, data.y)
        eval_loss += loss
    print('test :', accuracy / len(dataloader))

    return (accuracy / len(dataloader)), eval_loss