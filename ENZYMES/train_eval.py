import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np


def train(model, args, data: Data):
    device = torch.device(args.device)
    data = data.to(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    model.train()
    min_loss = 1e10
    for epoch in range(1, 1 + args.epochs):
        optimizer.zero_grad()
        out = model(data, args)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        test_loss, test_acc = val_evaluate(model, data, args)
        if test_loss < min_loss:
            torch.save(model.state_dict(), f'latest_{args.dataset}.pth')
            print("Model saved at epoch{}".format(epoch))
            min_loss = test_loss
        if epoch % 100 == 0:
            test_loss, test_acc = val_evaluate(model, data, args)
            print('epoch:{},loss:{},test loss:{},test accuracy:{}'.format(epoch, loss.data, test_loss, test_acc))
    model.load_state_dict(torch.load(f'latest_{args.dataset}.pth'))
    return model, loss


def val_evaluate(model, data: Data, args):
    model.eval()
    out = model(data, args)
    val_loss = F.nll_loss(out[data.test_mask], data.y[data.test_mask]).item()
    pred = out.max(dim=1)[1]
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    val_acc = int(correct) / int(data.test_mask.sum())
    return val_loss, val_acc


def evaluate(model, data: Data, args):
    model.eval()
    _, pred = model(data, args).max(dim=1)
    print('real:', data.y[data.test_mask])
    print('pred:', pred[data.test_mask])
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    return acc


def cross_validation(model):
    acc_list = list()
    for i in range(1, 6):
        train(model, i)
        acc_list.append(evaluate(model, i))
    mean_acc = np.mean(acc_list)
    return mean_acc, acc_list
