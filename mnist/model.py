import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN_MNIST(nn.Module):
    def __init__(self, n_capas=1, base_channels=32):
        super().__init__()

        self.blocks = nn.ModuleList()
        in_ch, out_ch = 3, base_channels  # 3 canales RGB
        for _ in range(n_capas):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            ))
            in_ch, out_ch = out_ch, out_ch * 2

        spatial  = 28 // (2 ** n_capas) #tama;o imagenes de MNIST es 28x28
        self.fc1 = nn.Linear(in_ch * spatial * spatial, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def conv_layers(self):
        return [block[0] for block in self.blocks]
