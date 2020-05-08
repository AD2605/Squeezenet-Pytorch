import torch
import torch.nn as nn


class Fire(nn.Module):
    def __init__(self,in_channels,squeeze_channels,k1_channels,k3_channels):
        super(Fire, self).__init__()

        self.in_channels = in_channels
        self.squeeze_channels = squeeze_channels
        self.k1_channels = k1_channels
        self.k3_channels = k3_channels

        self.squeeze_layer = self.get_squeeze_layer().cuda()
        self.expand1_layer = self.expand_1_layer().cuda()
        self.expand3_layer = self.expand_3_layer().cuda()

    def get_squeeze_layer(self):
        layers = []
        layers.append(nn.Conv2d(self.in_channels,self.squeeze_channels,kernel_size=1))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def expand_1_layer(self):
        layers = []
        layers.append(nn.Conv2d(self.squeeze_channels,self.k1_channels,kernel_size=1))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def expand_3_layer(self):
        layers = []
        layers.append(nn.Conv2d(self.squeeze_channels,self.k3_channels,kernel_size=3,padding=1))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.squeeze_layer(x)
        return torch.cat([self.expand1_layer(y),self.expand3_layer(y)], dim=1)

class SqueezeNet(nn.Module):
    def __init__(self,channels,classes ):
        super(SqueezeNet, self).__init__()

        self.channels = channels
        self.classes = classes

        self.conv1 = nn.Conv2d(self.channels, 64, kernel_size=3, stride=2)
        self.layers = []
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        self.layers.append(Fire(64, 16, 64, 64))
        self.layers.append(Fire(128, 16, 64, 64))
        self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        self.layers.append(Fire(128, 32, 128, 128))
        self.layers.append(Fire(256, 32, 128, 128))
        self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        self.layers.append(Fire(256, 48, 192, 192))
        self.layers.append(Fire(384, 48, 192, 192))
        self.layers.append(Fire(384, 64, 256, 256))
        self.layers.append(Fire(512, 64, 256, 256))
        self.layers.append(nn.Dropout())
        self.layers.append(nn.Conv2d(512, self.classes, kernel_size=1))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.AvgPool2d(13, stride=1))
        for layer in self.layers:
            layer.cuda()

    def forward(self, x):
        out = self.conv1(x)
        for layer in self.layers:
            out = layer(out)
        if self.classes ==2:
            return out.view(out.size(0), 1)
        else:
            return out.view(out.size(0), self.classes)

    def train_model(self, model, data, epochs):
        if self.classes ==2:
            criterion = nn.BCEWithLogitsLoss().cuda()
        else:
            criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10)
        min_loss = 5000
        model.train()
        model.cuda()
        for epoch in range(0, epochs):
            train_accuracy = 0
            net_loss = 0
            for _, (x, y) in enumerate(data):
                optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                max_index = out.max(dim=1)[1]
                accuracy = (max_index==y).sum()
                train_accuracy += accuracy.item()
                net_loss+=loss.item()

            scheduler.step()
            print('---------------------------------------------------------')
            print(epoch)
            print('AVERAGE LOSS = ', net_loss/len(data))
            print('TRAIN ACCURACY = ', train_accuracy/len(data))
            scheduler.step()
            if net_loss<min_loss:
                torch.save(model.state_dict(), '/save_path/squeezenet.pth')

    def evaluate(self, model, dataloader):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        correct = 0
        model.cuda()
        for _, (x, y) in enumerate(dataloader):
            x = x.cuda()
            y = y.cuda()
            out = model(x)
            if torch.argmax(out) == y:
                correct += 1
        print(correct / len(dataloader))
