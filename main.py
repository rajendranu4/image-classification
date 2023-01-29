import pandas as pd

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim

from ImageProcessing import *
from NetworkImage import *


def check_gpu():
    # checks if gpu is availbale or not for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device


if __name__ == '__main__':

    class_labels = ['cats', 'dogs']
    img_root_train_dir = 'dataset/train'
    print("Image Classification Using Logistic Regression with Image size - 64x64")
    img_process_train = ImageProcessingCLF()
    img_data_train = img_process_train.pre_processing(class_labels, img_root_train_dir, image_size=64)
    X_train, y_train = img_process_train.separate()

    img_root_test_dir = 'dataset/valid'
    img_process_test = ImageProcessingCLF()
    img_data_test = img_process_test.pre_processing(class_labels, img_root_test_dir, image_size=64)
    X_test, y_test = img_process_test.separate()

    print("Initializing model....")
    model_lr = LogisticRegressionCV(cv=5, solver='liblinear', random_state=0)
    print("Training model....")
    model_lr.fit(X_train, y_train)

    print("Testing model....")
    y_pred = model_lr.predict(X_test)

    print("\nTesting Accuracy Score:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix")
    conf_mat = confusion_matrix(y_test, y_pred)

    confusion_matrix_df = pd.DataFrame(conf_mat, columns=["Pred-->Cat", "Pred-->Dog"],
                                       index=["Orig-->Cat", "Orig-->Dog"])

    print(confusion_matrix_df)

    image_size_array = [64, 256]

    device = check_gpu()

    for size in image_size_array:
        print("\nImage Classification using Resnet model in Pytorch with image size - {}x{}".format(size, size))

        resnet_model = ResnetModel(device, image_size=size)

        image_classificaion = ImageClassificationPytorch(resnet_model, device)
        train_dataset, train_dataloader = image_classificaion.prepare(data_directory='dataset/train', transform=True,
                                                                      mode='train', batch_size=32, image_size=size)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(resnet_model.get_model().fc.parameters())

        n_epochs = 3
        image_classificaion.train(criterion, optimizer, n_epochs, train_dataloader, len(train_dataset))

        test_dataset, test_dataloader = image_classificaion.prepare(data_directory='dataset/valid', transform=True,
                                                                    mode='test', batch_size=-1)

        iter_loader = iter(test_dataloader)
        X_test, y_test = next(iter_loader)
        _, y_pred = image_classificaion.predict(X_test)

        print("\nTesting Accuracy Score:", accuracy_score(y_test.data, y_pred.data))