import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Define the Neural Network used in the reverse process
class DenoisingNN(nn.Module):
    def __init__(self):
        super(DenoisingNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 128),  # Reduced size
            nn.ReLU(),
            nn.Linear(128, 28 * 28)  # Output size is flattened image size
        )

    def forward(self, x):
        return self.fc(x)


# Forward process: adding noise to the data
def forward_process(x0, t, alpha_t):
    noise = torch.randn_like(x0)
    alpha_t = torch.tensor(alpha_t)  # Ensure alpha_t is a tensor
    xt = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise
    return xt, noise


# Reverse process: denoising the data
def reverse_process(xt, model, t, alpha_t):
    xt_reconstructed = model(xt)
    return xt_reconstructed


# Training the diffusion model
def train(model, optimizer, dataloader, num_steps=10):
    model.train()
    for step in range(num_steps):
        total_loss = 0
        for x0, _ in dataloader:
            x0 = x0.view(x0.size(0), -1)  # Flatten the images
            t = torch.tensor([0.1])  # Noise level
            alpha_t = 0.5  # Example alpha_t value

            xt, epsilon = forward_process(x0, t, alpha_t)
            optimizer.zero_grad()
            xt_reconstructed = reverse_process(xt, model, t, alpha_t)
            loss = torch.mean((xt_reconstructed - x0.view(xt_reconstructed.size())) ** 2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"Step {step}, Loss: {loss.item()}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {step}, Average Loss: {avg_loss}")


# Load the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize and train the model
model = DenoisingNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training started...")
train(model, optimizer, dataloader, num_steps=10)
print("Training completed.")


# Generate images using the trained model
def generate_images(model, num_images=5):
    model.eval()
    with torch.no_grad():
        noise = torch.randn(num_images, 28 * 28)  # Random noise
        t = torch.tensor([0.1])  # Noise level
        alpha_t = 0.5  # Example alpha_t value
        generated_images = reverse_process(noise, model, t, alpha_t)

        plt.figure(figsize=(10, 5))
        for i in range(num_images):
            plt.subplot(1, num_images, i + 1)
            plt.imshow(generated_images[i].view(28, 28).numpy(), cmap='gray')
            plt.axis('off')
        plt.show()


generate_images(model)
