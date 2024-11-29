import torch
import torch.nn as nn
import torchvision


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5, stride=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=1)
        self.conv2 = nn.Conv2d(5, 3, kernel_size=3, stride=2, padding=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv3 = nn.Conv2d(3, 2, kernel_size=2, stride=1)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.fc1 = nn.Linear(2508, 1254)
        self.fc2 = nn.Linear(1254, 627)
        self.fc3 = nn.Linear(627, 32)


        # layer's output dimension must match number of classes in dataset
        self.fc4 = nn.Linear(64, 4)

    def __forward_once(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))

        return x

    def forward(self, xs):
        # unpack pairs
        xs_1, xs_2 = torch.unsqueeze(xs[:, 0, :, :], 1), torch.unsqueeze(xs[:, 1, :, :], 1)
        ys_1, ys_2 = self.__forward_once(xs_1), self.__forward_once(xs_2)
        return self.fc4(torch.cat((ys_1, ys_2), dim=1))

    def _forward(self, xs_1, xs_2, ys_1, ys_2):
        xs_1, xs_2 = xs_1, xs_2
        ys_1, ys_2 = ys_1, ys_2

        #ys_1, ys_2 = self.__forward_once(xs_1), self.__forward_once(xs_2)
        return self.fc4(torch.cat((ys_1,ys_2),dim=1))



class ResNet18(nn.Module):
    """
        Siamese network for image similarity estimation.
        The network is composed of two identical networks, one for each input.
        The output of each network is concatenated and passed to a linear layer.
        The output of the linear layer passed through a sigmoid function.
        `"FaceNet" <https://arxiv.org/pdf/1503.03832.pdf>`_ is a variant of the Siamese network.
        This implementation varies from FaceNet as we use the `ResNet-18` model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_ as our feature extractor.
        In addition, we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    """

    def __init__(self):
        super(ResNet18, self).__init__()
        # get resnet model
        self.resnet = torchvision.models.resnet18(weights=None)

        # over-write the first conv layer to be able to read MNIST images
        # as resnet18 reads (3,x,x) where 3 is RGB channels
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features

        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
        )

        # initialize the weights
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, xs):
        input1, input2 = torch.unsqueeze(xs[:, 0, :, :], 1), torch.unsqueeze(xs[:, 1, :, :], 1)
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        return output
