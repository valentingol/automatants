# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 18:18:51 2021

@author: valentin
"""
#%%
import torch 
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F

#%%

x = torch.tensor(3., requires_grad=True)
y = torch.tensor(3., requires_grad=True)

def f(x, y):
    return x**2 + y**2

lr = 0.1

for _ in range(100):
    res = f(x, y)
    res.backward()
    with torch.no_grad():
        x -= lr * x.grad
        y -= lr * y.grad
    x.grad.zero_(), y.grad.zero_()

print(f'x = {x}\ny = {y}\nres = {f(x, y)}')

#%%
from torchvision import datasets, transforms

transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(0.1307, 0.3081)])

train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data',
                                                     train=True,
                                                     download=True,
                                                     transform=transforms),
                                           shuffle=True,
                                           batch_size=32)

test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data',
                                                     train=False,
                                                     download=True,
                                                     transform=transforms),
                                           shuffle=True,
                                           batch_size=32)

for data_batch, label_batch in train_loader:
    fig = plt.figure()
    for i in range(1, 31):
        img = data_batch[i, 0]
        label = label_batch[i]
        plt.subplot(5, 6, i)
        plt.axis('off')
        plt.title(f'{label}')
        plt.imshow(img, cmap='gray_r')
    plt.show()
    break
#%%

N_EPOCHS = 13

x_axis, y_axis1, y_axis2 = list(range(1, N_EPOCHS+1)), [], []

#%%

class Dense(nn.Module):
    def __init__(self):
        super(Dense, self).__init__()
        self.layer1 = nn.Linear(28*28, 64)
        self.layer2 = nn.Linear(64, 10)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.ndim > 1:
                p = nn.init.kaiming_normal_(p)
            else:
                p = nn.init.normal_(p)

    def forward(self, x):
        x = x.view((-1, 28*28))
        x = F.relu(self.layer1(x))
        x = F.softmax(self.layer2(x), -1)
        return x

model = Dense()

learning_rate = 1e-3
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def acc(pred, labels):
    classes_pred = torch.argmax(pred, dim=-1)
    n_correct = 0
    for i, class_pred in enumerate(classes_pred):
        if class_pred == labels[i]:
            n_correct += 1
    return n_correct/(i+1)

for epoch in range(N_EPOCHS):
    print(f'epoch {epoch+1}/{N_EPOCHS}')
    for n_batch, (data_batch, label_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        pred = model(data_batch)
        loss = criterion(pred, label_batch)
        loss.backward()
        optimizer.step()
        train_acc = acc(pred, label_batch)
        print(f'\r batch {n_batch+1}/{60000//32}, '
              f'loss = {float(loss) : .4}  '
              f'acc = {train_acc : 4.3}  ', end='')
        
    print()
    loss, valid_acc = 0., 0.
    for n_batch, (data_batch, label_batch) in enumerate(test_loader):
        data_batch = data_batch.view((-1, 28*28))
        with torch.no_grad():
            pred = model(data_batch)
            loss += criterion(pred, label_batch)
            valid_acc += acc(pred, label_batch)
    valid_acc /= (n_batch+1)
    loss /= (n_batch+1)
    y_axis1.append(valid_acc)
    print(f' valid_loss = {loss : .4}  \n '
          f'valid_acc = {valid_acc : 4.3}  ')
    
    print()

#%%

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dense1 = nn.Linear(320, 50)
        self.dense2 = nn.Linear(50, 10)
        self.init_weights
    
    def init_weights(self):
        for p in self.parameters():
            if p.ndim > 1:
                p = nn.init.kaiming_normal_(p)
            else:
                p = nn.init.normal_(p)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.dropout2d(self.conv2(x), training=self.training)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self.dense1(x))
        x = F.dropout(x, training=self.training)
        x = F.softmax(self.dense2(x), -1)
        return x 

model = CNN()

learning_rate = 1e-3
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def acc(pred, labels):
    classes_pred = torch.argmax(pred, dim=-1)
    n_correct = 0
    for i, class_pred in enumerate(classes_pred):
        if class_pred == labels[i]:
            n_correct += 1
    return n_correct/(i+1)

for epoch in range(N_EPOCHS):
    print(f'epoch {epoch+1}/{N_EPOCHS}')
    model.training = True
    for n_batch, (data_batch, label_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        pred = model(data_batch)
        loss = criterion(pred, label_batch)
        loss.backward()
        optimizer.step()
        train_acc = acc(pred, label_batch)
        print(f'\r batch {n_batch+1}/{60000//32}, '
              f'loss = {float(loss) : .4}  '
              f'acc = {train_acc : 4.3}  ', end='')
    print()
    loss, valid_acc = 0., 0.
    model.training = False
    for n_batch, (data_batch, label_batch) in enumerate(test_loader):
        with torch.no_grad():
            pred = model(data_batch)
            loss += criterion(pred, label_batch)
            valid_acc += acc(pred, label_batch)
    valid_acc /= (n_batch+1)
    loss /= (n_batch+1)
    y_axis2.append(valid_acc)
    print(f' valid_loss = {loss : .4}   \n '
          f'valid_acc = {valid_acc : 4.3}  ')
    print()

#%%

fig = plt.figure()
plt.title('valid accuracy')
ax = plt.gca()
ax.set_ylim([0.93, 0.98])
plt.plot(x_axis, y_axis1, 'b', label='dense')
plt.plot(x_axis, y_axis2, 'r', label='conv')
plt.legend(loc='upper left')
plt.show()
