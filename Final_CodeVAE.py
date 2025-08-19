import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

# Define hyperparameters
batch_size = 128
learning_rate = 1e-3
num_epochs = 10
latent_dim = 20

# Load MNIST dataset 
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

#note : transforms.Tensor ()convert the data from 0 to 1 range
#       becasue of that we use the cross entropy loss

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define the encoder network
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1) # ouput size 14x14x 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # output size 7x7x64
        
        self.fc1 = nn.Linear(7  * 7 * 64, 256) #
        self.fc2_mean = nn.Linear(256, latent_dim)
        self.fc2_logvar = nn.Linear(256, latent_dim)

    def forward(self, x): 
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # view the tensor as single dimensional like flatten it
        x = torch.relu(self.fc1(x))
        z_mean = self.fc2_mean(x)
        z_logvar = self.fc2_logvar(x)
        return z_mean, z_logvar

# Define the decoder network
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 7 * 7 * 64)
        self.conv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = x.view(x.size(0), 64, 7, 7) # view the tensor as a single dimensional vector
        x = torch.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x)) # sigmoid output 786 value of range [0 to 1] for reconstruction image 
        return x

# Define the VAE model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) #=> logvar = log(std^2)=> output of variance vector
                                      #=> std = exp(0.5 * logvar) => inorder to make it positve and negative values for a stable training. 
        eps = torch.randn_like(std) # Returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1
        return mu + eps * std

    def forward(self, x):
        z_mean, z_logvar = self.encoder(x)
        z = self.reparameterize(z_mean, z_logvar)
        x_recon = self.decoder(z)
        return x_recon, z_mean, z_logvar

# Instantiate the VAE model and define the loss function and optimizer
model = VAE()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#

# Define the loss function
def loss_fn(x_recon, x, z_mean, z_logvar):
    recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum') # reduction = sum means output will be summed 
    kl_div = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    return recon_loss + kl_div

# Train the model
train_loss = []
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for i, (images, _) in enumerate(train_loader):  # as we dont use label so put _ over there
        # Forward pass
        x = Variable(images) # Autograd automatically supports
                             #Tensors with requires_grad set to True.
                             #The original purpose of Variables was to be able to use automatic differentiation (Source):

        x_recon, z_mean, z_logvar = model(x)
        # Compute loss
        loss = loss_fn(x_recon, x, z_mean, z_logvar)
        epoch_loss += loss.item()
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_loss /= len(train_loader.dataset)
    train_loss.append(epoch_loss)
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss))

# Test the model
model.eval()
test_loss = 0.0
with torch.no_grad():
    for images, _ in test_loader:
        x = Variable(images)
        x_recon, z_mean, z_logvar = model(x)
        loss = loss_fn(x_recon, x, z_mean, z_logvar)
        test_loss += loss.item()
test_loss /= len(test_loader.dataset)
print('Test Loss: {:.4f}'.format(test_loss))

# Generate new images
sample = Variable(torch.randn(64, latent_dim))
sample = model.decoder(sample)
sample = sample.view(64, 1, 28, 28).data
plt.figure(figsize=(8, 8))
for i in range(64):
    plt.subplot(8, 8, i+1)
    plt.imshow(sample[i][0], cmap='gray')
    plt.axis('off')
plt.show()
