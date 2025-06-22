#-----------TRAINING COLAB FILE CODE----------(CODE FOR THE APP IS BELOW THIS CODE)
#-----------comment training code then run app code or visa versa

# GAN Training for MNIST Digit Generation
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# Save model to Google Drive
SAVE_DIR = '/content/drive/MyDrive/handwriting'
os.makedirs(SAVE_DIR, exist_ok=True)

# Define Generator
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )
        self.img_shape = img_shape

    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), *self.img_shape)

# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28)):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(torch.prod(torch.tensor(img_shape))), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
batch_size = 64
epochs = 20
img_shape = (1, 28, 28)

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
train_loader = DataLoader(
    datasets.MNIST('/content/data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)


# Models
generator = Generator(latent_dim, img_shape).to(device)
discriminator = Discriminator(img_shape).to(device)

# Loss and Optimizers
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

# Training loop
for epoch in range(epochs):
    g_epoch_loss = 0.0
    d_epoch_loss = 0.0
    for i, (imgs, _) in enumerate(train_loader):
        real_imgs = imgs.to(device)
        valid = torch.ones(imgs.size(0), 1, device=device)
        fake = torch.zeros(imgs.size(0), 1, device=device)

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(imgs.size(0), latent_dim, device=device)
        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        g_epoch_loss += g_loss.item()
        d_epoch_loss += d_loss.item()

        if (i + 1) % 200 == 0:
            print(f"[Epoch {epoch+1}/{epochs}] [Batch {i+1}/{len(train_loader)}] D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    # Epoch summary
    print(f"Epoch {epoch+1} completed. Avg D Loss: {d_epoch_loss/len(train_loader):.4f}, Avg G Loss: {g_epoch_loss/len(train_loader):.4f}")

    # Show generated image sample
    with torch.no_grad():
        z = torch.randn(5, latent_dim, device=device)
        samples = generator(z).detach().cpu()
        samples = samples * 0.5 + 0.5  # Unnormalize from [-1, 1] to [0, 1]

        fig, axs = plt.subplots(1, 5, figsize=(10, 2))
        for k in range(5):
            axs[k].imshow(samples[k][0], cmap='gray')
            axs[k].axis('off')
        plt.suptitle(f"Sample Generated Images (Epoch {epoch+1})")
        plt.show()

# Save Generator
torch.save(generator.state_dict(), os.path.join(SAVE_DIR, "generator.pth"))
print(f"Generator model saved to: {os.path.join(SAVE_DIR, 'generator.pth')}")






# --------------APP FILE ----------------------
import streamlit as st
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from PIL import Image
import numpy as np
import os

# Load trained model
class Generator(torch.nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, int(np.prod(img_shape))),
            torch.nn.Tanh()
        )
        self.img_shape = img_shape

    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), *self.img_shape)

# Config
LATENT_DIM = 100
IMG_SHAPE = (1, 28, 28)

@st.cache_resource
def load_generator():
    model = Generator(LATENT_DIM, IMG_SHAPE)
    model.load_state_dict(torch.load("generator.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Web UI
st.title("Handwritten Digit Generator (0–9)")
digit = st.selectbox("Choose a digit to generate:", list(range(10)))

if st.button("Generate 5 Images"):
    generator = load_generator()
    z = torch.randn(5, LATENT_DIM)
    with torch.no_grad():
        imgs = generator(z).cpu() * 0.5 + 0.5  # Rescale [-1,1] to [0,1]

    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axes[i].imshow(imgs[i][0], cmap='gray')
        axes[i].axis('off')
    st.pyplot(fig)
