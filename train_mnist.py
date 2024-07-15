import os
import pytz
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from model.diffuser import Diffuser
from model.unet import UNet

img_size = 28
batch_size = 1024
num_timesteps = 1000
epochs = 100
lr = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

jst = pytz.timezone("Asia/Tokyo")
start_date = datetime.strftime(datetime.now(jst), "%Y%m%d_%H%M")
log_dir = f"./logs/{start_date}"
resume_from_checkpoint = "./logs/20240703_1153/best_model.pth"
best_model_path = f"{log_dir}/best_model.pth"

os.makedirs(log_dir, exist_ok=True)


def save_images(images, save_dir, rows=2, cols=10):
    fig = plt.figure(figsize=(cols, rows))
    i = 0
    for _ in range(rows):
        for _ in range(cols):
            fig.add_subplot(rows, cols, i + 1)
            plt.imshow(images[i], cmap="gray")
            plt.axis("off")
            i += 1
    plt.savefig(f"{save_dir}/generated.png")


preprocess = transforms.ToTensor()
dataset = torchvision.datasets.MNIST(root="./data", download=True, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

diffuser = Diffuser(num_timesteps, device=device)
model = UNet()
optimizer = Adam(model.parameters(), lr=lr)

if resume_from_checkpoint != "":
    cp = torch.load(resume_from_checkpoint)
    model.load_state_dict(cp)

model.to(device)

losses = []
best_loss = float("inf")

for epoch in range(epochs):
    loss_sum = 0.0
    cnt = 0

    for images, labels in tqdm(dataloader):
        optimizer.zero_grad()
        x = images.to(device)
        t = torch.randint(1, num_timesteps + 1, (len(x),), device=device)

        x_noisy, noise = diffuser.add_noise(x, t)
        noise_pred = model(x_noisy, t)
        loss = F.mse_loss(noise, noise_pred)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        cnt += 1

    loss_avg = loss_sum / cnt
    losses.append(loss_avg)
    print(f"Epoch {epoch} | Loss: {loss_avg}")

    if loss_avg < best_loss:
        best_loss = loss_avg
        torch.save(model.state_dict(), best_model_path)


plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(f"{log_dir}/loss.png")

images = diffuser.sample(model)
save_images(images, log_dir)
