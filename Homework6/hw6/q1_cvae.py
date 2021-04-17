# EECS545 HW6: CVAE.

import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


class CVAE(nn.Module):

    def __init__(self, input_size, latent_size, class_size):
        super(CVAE, self).__init__()
        self.input_size = input_size
        self.class_size = class_size
        self.latent_size = latent_size
        self.units = 400

        ######################################################
        ###              START OF YOUR CODE                ###
        ######################################################
        ### Define a three layer neural network architecture #
        ### for the recognition_model                        #
        ######################################################
        # TODO: Implement Here.
        self.fc1 = nn.Linear(input_size + class_size, self.units)
        self.fc2 = nn.Linear(self.units, self.units)
        self.fc3 = nn.Linear(self.units, self.units)
        self.mu = nn.Linear(self.units, self.latent_size)
        self.logvar = nn.Linear(self.units, self.latent_size)

        ######################################################
        ###               END OF YOUR CODE                 ###
        ######################################################


        ######################################################
        ###              START OF YOUR CODE                ###
        ######################################################
        ### Define a three layer neural network architecture #
        ### for the generation_model                         #
        ######################################################
        # TODO: Implement Here.
        self.fc4 = nn.Linear(latent_size + class_size, self.units)
        self.fc5 = nn.Linear(self.units, self.units)
        self.fc6 = nn.Linear(self.units, self.units)
        self.output_layer = nn.Linear(self.units, input_size)

        ######################################################
        ###               END OF YOUR CODE                 ###
        ######################################################



    def recognition_model(self, x, c):
        """
        Computes the parameters of the posterior distribution q(z | x, c) using the
        recognition network defined in the constructor

        Inputs:
        - x: PyTorch Variable of shape (batch_size, input_size) for the input data
        - c: PyTorch Variable of shape (batch_size, num_classes) for the input data class

        Returns:
        - mu: PyTorch Variable of shape (batch_size, latent_size) for the posterior mu
        - logvar PyTorch Variable of shape (batch_size, latent_size) for the posterior
          variance in log space
        """
        ###########################
        # TODO: Implement Here.
        ###########################
        mu = None
        logvar = None

        relu = nn.ReLU()
        x_c = torch.cat([x, c], -1)
        out11 = self.fc1(x_c)
        out12 = relu(out11)
        out21 = self.fc2(out12)
        out22 = relu(out21)
        out31 = self.fc3(out22)
        out32 = relu(out31) 
        mu = self.mu(out32)
        logvar = self.logvar(out32)

        return mu, logvar


    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std) + mu

    def generation_model(self, z, c): # P(x|z, c)
        """
        Computes the generation output from the generative distribution p(x | z, c)
        using the generation network defined in the constructor

        Inputs:
        - z: PyTorch Variable of shape (batch_size, latent_size) for the latent vector
        - c: PyTorch Variable of shape (batch_size, num_classes) for the input data class

        Returns:
        - x_hat: PyTorch Variable of shape (batch_size, input_size) for the generated data
        """
        ###########################
        # TODO: Implement Here.
        ###########################
        x_hat = None
        relu = nn.ReLU()
        sigmoid = nn.Sigmoid()
        z_c = torch.cat([z,c], -1)
        out41 = self.fc4(z_c)
        out42 = relu(out41)
        out51 = self.fc5(out42)
        out52 = relu(out51)
        out61 = self.fc6(out52)
        out62 = relu(out61)
        output = self.output_layer(out62)
        x_hat = sigmoid(output)

        return x_hat

    def forward(self, x, c):
        """
        Performs the inference and generation steps of the CVAE model using
        the recognition_model, reparametrization trick, and generation_model

        Inputs:
        - x: PyTorch Variable of shape (batch_size, input_size) for the input data
        - c: PyTorch Variable of shape (batch_size, num_classes) for the input data class

        Returns:
        - x_hat: PyTorch Variable of shape (batch_size, input_size) for the generated data
        - mu: PyTorch Variable of shape (batch_size, latent_size) for the posterior mu
        - logvar: PyTorch Variable of shape (batch_size, latent_size)
                  for the posterior logvar
        """
        ###########################
        # TODO: Implement Here.
        ###########################
        x_hat = None
        mu = None
        logvar = None
        x = x.view(-1, self.input_size)

        mu, logvar = self.recognition_model(x, c)
        z = self.reparametrize(mu, logvar)
        x_hat = self.generation_model(z, c)


        return x_hat, mu, logvar


def to_var(x, use_cuda):
    x = Variable(x)
    if use_cuda:
        x = x.cuda()
    return x


def one_hot(labels, class_size, use_cuda):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return to_var(targets, use_cuda)


def train(epoch, model, train_loader, optimizer, num_classes, use_cuda):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = to_var(data, use_cuda).view(data.shape[0], -1)
        labels = one_hot(labels, num_classes, use_cuda)
        recon_batch, mu, logvar = model(data, labels)
        optimizer.zero_grad()
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data))


def loss_function(x_hat, x, mu, logvar):
    """
    Computes the negative variational lowerbound averaged over the minibatch for conditional vae
    Note: We compute -lowerbound because we optimize the network by minimizing a loss

    Inputs:
    - x_hat: PyTorch Variable of shape (batch_size, input_size) for the generated data
    - x: PyTorch Variable of shape (batch_size, input_size) for the real data
    - mu: PyTorch Variable of shape (batch_size, latent_size) for the posterior mu
    - logvar: PyTorch Variable of shape (batch_size, latent_size) for the posterior logvar

    Returns:
    - loss: PyTorch Variable containing the (scalar) loss for the negative lowerbound.
    """
    ###########################
    # TODO: Implement Here.
    ###########################
    loss = None
    binary_cross_entropy = F.binary_cross_entropy(x_hat, x, size_average=False)
    DKL = -torch.sum(1 + logvar - mu**2 - torch.exp(logvar))/2
    loss = (binary_cross_entropy + DKL)/x.shape[0]
    return loss


use_cuda = False
input_size = 28 * 28
units = 400
batch_size = 32
latent_size = 20 # z dim
num_classes = 10
num_epochs = 10


def main():
    # Load MNIST dataset
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    dataset = datasets.MNIST(
        './data',  train=True, download=True,
        transform=transforms.ToTensor())
    train_dataset = torch.utils.data.Subset(dataset, indices=range(10000))
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True, **kwargs)

    model = CVAE(input_size, latent_size, num_classes)

    if use_cuda:
        model.cuda()

    # Note: You will get an ValueError here if you haven't implemented anything
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    start = time.time()
    for epoch in range(1, num_epochs+1):
        train(epoch, model, train_loader, optimizer, num_classes, use_cuda)
    print('training time = %f'%(time.time() - start)) # should take less than 5 minutes


    # Generate images with condition labels
    c = torch.eye(num_classes, num_classes) # [one hot labels for 0-9]
    c = to_var(c, use_cuda)
    z = to_var(torch.randn(num_classes, latent_size), use_cuda)
    samples = model.generation_model(z, c).data.cpu().numpy()

    fig = plt.figure(figsize=(10, 1))
    gs = gridspec.GridSpec(1, 10)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    plt.show()


if __name__ == '__main__':
    main()
