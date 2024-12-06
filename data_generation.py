import numpy as np
import torch
import torch.nn.functional as F
import torch
import pandas as pd

def generate_data(randomData):
    if randomData:
        distrib = torch.distributions.Bernoulli(0.7)
        board = distrib.sample((32, 32)).view(1,1,32,32)
    else:
        df = pd.read_csv("Training_Board.csv", header=None)
        board = torch.tensor(df.values, dtype=torch.int64).view(1, 1, df.shape[0], df.shape[1])


    weights = torch.tensor([[1,1,1],[1,10,1],[1,1,1]]).view(1,1,3,3)


    data = [board]

    for _ in range(255):
        newboard = F.conv2d(board, weights, padding=1).view(32, 32)
        newboard = (newboard==12) | (newboard==3) | (newboard==13)
        newboard_array = np.int8(newboard) * 255 
        board = torch.tensor(newboard_array/255, dtype=torch.int64).view(1,1,32,32)
        data.append(board)
        print(len(data))

    torch.save(data, 'train.data')


    last_frame = None
    newdata = []
    for i, d in enumerate(data):
        if i == 0:
            last_frame = d
            continue
        if abs(d.sum().item()) > 1.5 * abs(last_frame.sum().item()):
            pass
        else:
            newdata.append((last_frame, d))
        last_frame = d

    torch.save(newdata, 'train.data.batch')
    print("Data saved")


