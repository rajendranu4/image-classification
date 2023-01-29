import torch
import torch.nn as nn
from torchvision import datasets, models, transforms


class ResnetModel(nn.Module):
    def __init__(self, device, image_size=256):
        super().__init__()
        print("Initializing model....")
        self.device = device

        # pretrained is true to import the pretrained resnet model with various images from various sources
        self.model = models.resnet50(pretrained=True).to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False

        # adding a fully connected layer which serves as the classifier for our problem
        self.model.fc = nn.Sequential(
            nn.Linear(2048, image_size),
            nn.ReLU(inplace=True),
            nn.Linear(image_size, 2)).to(device)

    def get_model(self):
        return self.model

    def print_model(self):
        print("Model specification")
        print(self.model)

    def forward(self, x):
        # inputs are fed forwarded to produce output
        x = self.model(x)
        # x = self.fc(x)

        return x


class ImageClassificationPytorch:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def prepare(self, data_directory, mode, batch_size, transform=False, image_size=256):
        print("Preparing data....")

        # transforms an image according to specifications in the transform function
        if transform:
            if mode == 'train':
                data_transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
            else:
                data_transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor()
                ])
        else:
            data_transform = None

        dataset = datasets.ImageFolder(data_directory, data_transform)

        if batch_size == -1:
            batch_size = len(dataset)

        # creates data loader with batch size
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        return dataset, dataloader

    def train(self, criterion, optimizer, n_epochs, train_dataloader, train_size):
        print("Training model....")
        for epoch in range(n_epochs):

            epoch_loss_sum = 0.0
            self.model.train()

            # inputs are fed forwarded to produce output
            # loss is calculated witht the predicted and original outputs
            # gradients or delta are calculated with respect to the loss
            # delta loss is back propogated backwards to adjust weights and bias
            for X_data, y_original in train_dataloader:
                X_data = X_data.to(self.device)
                y_original = y_original.to(self.device)

                y_predicted = self.model.forward(X_data)
                loss = criterion(y_predicted, y_original)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss_sum += loss.item() * X_data.size(0)

            print('Epoch {} of {} ==> loss: {:.3f}'.format(epoch + 1, n_epochs, epoch_loss_sum / train_size))

    def predict(self, X_data):
        print("Testing model")

        # test instances are fed forwarded to produce output and the max of output probability
        # is chosen as the class label for a particular test instance
        return torch.max(self.model.forward(X_data), 1)