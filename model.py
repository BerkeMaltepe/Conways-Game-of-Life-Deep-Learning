import torch
import numpy as np
import torch.nn.functional as F
import os.path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from data_generation import generate_data
from data_visualization import plot_losses

class GameOfLifeModel(torch.nn.Module):
    def __init__(self):
        super(GameOfLifeModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, (3, 3), padding=1)
        self.conv2 = torch.nn.Conv2d(1, 1, (3, 3), padding=1)
        self.fc1 = torch.nn.Linear(32 * 32, 32 * 32)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = torch.tanh(x)
        return x.view(-1, 1, 32, 32)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def test(data, model, loss_fn):
    model.eval()
    with torch.no_grad():
        output = model(data)
        loss = loss_fn(output, data)
        print(f"Test Loss: {loss.item()}")

def train(dataloader, model, loss_fn, optimizer, epochs):
    losses = []
    size = len(dataloader.dataset)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            last_frame, target_ = data
            target_ = target_.view(-1, 1, 32, 32).to(torch.float32).to(device)
            last_frame = last_frame.view(-1, 1, 32, 32).to(torch.float32).to(device)
            input_ = last_frame
        
            optimizer.zero_grad()

            output_ = model(input_)
            loss = loss_fn(output_, target_) * 56  # loss on non-aliver is penalized more
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 20 == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0

        losses.append(running_loss/size)
        torch.save(model.state_dict(), f"results/gameoflife_epoch_{epoch + 1}.pth")
    return losses

def evaluate(dataloader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            last_frame, target_ = data
            target_ = target_.view(-1, 1, 32, 32).to(torch.float32).to(device)
            last_frame = last_frame.view(-1, 1, 32, 32).to(torch.float32).to(device)
            input_ = last_frame

            output_ = model(input_)
            predicted = (output_ > 0.5).float()  # Binarize the output
            total += target_.numel()
            correct += (predicted == target_).sum().item()

    accuracy = correct / total
    print(f'Accuracy: {accuracy * 100:.2f}%')

def visualize_game(model, initial_frame, steps):
    model.eval()
    frames = [initial_frame]
    current_frame = initial_frame.to(device)

    with torch.no_grad():
        for _ in range(steps):
            output = model(current_frame)
            current_frame = (output > 0.5).float()  # Binarize the output
            frames.append(current_frame.cpu())

    fig = plt.figure()
    ims = [[plt.imshow(frame.squeeze().numpy(), cmap='gray', animated=True)] for frame in frames]

    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=1000)
    plt.show()

if __name__ == "__main__":
    if not os.path.exists('results'):
        os.makedirs('results')

    if not os.path.exists("train.data.batch"):
        generate_data(False)

    data = torch.load("train.data.batch")

    dataloader = torch.utils.data.DataLoader(data, batch_size=8, shuffle=True, num_workers=1)

    model = GameOfLifeModel().to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses = train(dataloader, model, loss_fn, optimizer, 35)
    evaluate(dataloader, model)
    plot_losses(losses, "GameOfLifeModel")
    initial_frame = data[0][0].view(1, 1, 32, 32).to(torch.float32)
    visualize_game(model, initial_frame, steps=50)
