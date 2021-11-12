from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms

def get_MNIST(batch_size: int=64, shuffle: bool=True):
    """ Return train and validation DataLoader from MNIST
    Download MNIST data if not downloaded yet. """
    transfos = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(0.1307, 0.3081)])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transfos),
        shuffle=shuffle, batch_size=batch_size
        )
    validation_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True,
                       transform=transfos),
        shuffle=shuffle, batch_size=batch_size
        )
    return train_loader, validation_loader


def plot_data(data_loader: torch.utils.data.DataLoader, n_imgs: int=30):
    """ Plot and display n_imgs binary images from data loader
    Note: n_imgs must be divided by 5 and lower than the
    batch_size of data_loader (if any). """
    if n_imgs > len(data_loader) or n_imgs % 5 != 0:
        raise ValueError('n_imgs must be divided by 5 and lower than the '
                         'batch_size of data_loader (if any).')
    for data_batch, label_batch in data_loader:
        plt.figure()
        for i in range(1, n_imgs + 1):
            img = data_batch[i, 0]
            label = label_batch[i]
            plt.subplot(5, n_imgs//5, i)
            plt.axis('off')
            plt.title(f'{label}')
            plt.imshow(img, cmap='gray_r')
        break
    plt.show()


def tensor_to_np(tensor: torch.Tensor) -> np.ndarray:
    """ Returns a numpy tensor from a torch tensor (in VRAM or not). """
    return tensor.detach().cpu().numpy()


class MLP(nn.Module):
    def __init__(self, name='mlp'):
        super(MLP, self).__init__()
        self.name = name
        self.layer1 = nn.Linear(28*28, 64)
        self.layer2 = nn.Linear(64, 10)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.ndim > 1:
                # Weights except biases
                p = nn.init.kaiming_normal_(p)
            else:
                # Biases
                p = nn.init.normal_(p)

    def forward(self, x):
        # Force flatten input if not already
        x = x.view((-1, 28*28))
        x = F.relu(self.layer1(x))
        # 10 outputs between 0 and 1 and sum to 1:
        x = F.softmax(self.layer2(x), -1)
        return x


class CNN(nn.Module):
    def __init__(self, name='cnn'):
        super(CNN, self).__init__()
        self.name = name
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dense1 = nn.Linear(320, 50)
        self.dense2 = nn.Linear(50, 10)
        self._init_weights

    def _init_weights(self):
        for p in self.parameters():
            if p.ndim > 1:
                # Weights except bias
                p = nn.init.kaiming_normal_(p)
            else:
                # Biases
                p = nn.init.normal_(p)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # Dropout to fight overfitting
        x = F.dropout2d(self.conv2(x), training=self.training)
        x = F.relu(F.max_pool2d(x, 2))
        # Flatten to compute dense layers of the classifier
        x = x.view(-1, 320)
        x = F.relu(self.dense1(x))
        x = F.dropout(x, training=self.training)
        # 10 outputs between 0 and 1 and sum to 1:
        x = F.softmax(self.dense2(x), -1)
        return x


def acc(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    good_preds = torch.argmax(pred, dim=-1) == labels
    return good_preds.float().mean()


def set_up_training(model, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return criterion, optimizer


def train_valid_loop(model, train_loader, validation_loader, device,
                     learning_rate, n_epochs=10):
    criterion, optimizer = set_up_training(model, learning_rate)

    train_acc_list, valid_acc_list = [], []
    for epoch in range(n_epochs):
        print(f'Epoch {epoch+1}/{n_epochs}')
        # Train loop
        model.training = True # change dropout and normalization behavior
        train_acc_mean = 0.0
        for batch, (data_batch, label_batch) in enumerate(train_loader):
            data_batch = data_batch.to(device)
            label_batch = label_batch.to(device)
            # Backpropagation
            optimizer.zero_grad()
            pred = model(data_batch)
            loss = criterion(pred, label_batch)
            loss.backward()
            optimizer.step()
            # Avoid computing gradients during metric calculation
            with torch.no_grad():
                acc_np = tensor_to_np(acc(pred, label_batch))
                train_acc_mean = (train_acc_mean * batch / (batch+1)
                                + acc_np / (batch+1))

            print(f' batch {batch}/{60000//batch_size}, '
                f'loss = {tensor_to_np(loss): .4f}  '
                f'acc = {train_acc_mean : 4.3f}  ', end='\r')
        train_acc_list.append(train_acc_mean)
        print()

        # Validation loop
        model.training = False # change dropout and normalization behavior
        valid_loss_mean = valid_acc_mean = 0.0
        for batch, (data_batch, label_batch) in enumerate(validation_loader):
            data_batch = data_batch.to(device)
            label_batch = label_batch.to(device)
            # Avoid computing gradients
            with torch.no_grad():
                pred = model(data_batch)
                loss_np = tensor_to_np(criterion(pred, label_batch))
                acc_np = tensor_to_np(acc(pred, label_batch))
                valid_loss_mean = (valid_loss_mean * batch / (batch+1)
                                + loss_np / (batch+1))
                valid_acc_mean = (valid_acc_mean * batch / (batch+1)
                                + acc_np / (batch+1))

        print(f' valid loss = {valid_loss_mean : .4}\n'
            f' valid acc = {valid_acc_mean : 4.3}')
        valid_acc_list.append(valid_acc_mean)
    return train_acc_list, valid_acc_list


def plot_accuracies(train_acc_list, valid_acc_list, color='b', lab='acc',
                    y_lim=[0.0, 1.0], title=None):
    """ Plot the training and validation accuracies (no displaying). """
    if title is None:
        plt.title('Traning and validation accuracy.', fontsize=12)
    plt.plot(train_acc_list, c=color, linestyle='dashed', label=lab + ' train')
    plt.plot(valid_acc_list, c=color, label=lab + ' valid')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(y_lim)
    plt.legend(loc='best')


if __name__ == '__main__':
    # Configs
    # device should be 'cpu' if cuda is not available. To check it, run:
    #  >>> torch.cuda.is_available()
    device = 'cuda'
    batch_size = 512
    n_epochs = 13
    learning_rate = 1e-3

    # Get data
    train_loader, validation_loader = get_MNIST(batch_size=batch_size,
                                                shuffle=True)
    # Show data examples
    plot_data(train_loader, n_imgs=30)

    # Multi-Layer-Perceptron model
    mlp = MLP(name='mlp').to(device)
    # CNN model
    cnn = CNN(name='cnn').to(device)

    # Train both models and plot the accuracies
    plt.figure()
    for model in [mlp, cnn]:
        train_acc_list, valid_acc_list = train_valid_loop(
            model, train_loader, validation_loader, device,
            learning_rate, n_epochs=n_epochs
            )
        color = '#0061c9' if model.name == 'mlp' else '#e38100'
        plot_accuracies(train_acc_list, valid_acc_list, color=color,
                        lab=model.name.upper(), y_lim=[0.8, 1.0])

    # Show the accuracy plots
    plt.show()
