
import os
import numpy as np
from numpy import asarray

from PIL import Image


class ImageProcessingCLF:
    def __init__(self):
        self.img_data = {}

    def pre_processing(self, class_labels, path, image_size=8):
        print("Preparing data....")
        labels = []
        data_array = []
        label_class = []
        label_track = 0
        # since we have two different folders for cats and dogs
        for label in class_labels:
            # print("\nDirectory")
            # print(label)

            labels.append({label: label_track})

            directory = path + '\\' + label

            # print(directory)
            # for every image file in the particular directory, it creates the image array and stores it
            # with the class label of that image
            for file in os.listdir(directory):
                filename = directory + '\\' + file
                image = Image.open(filename)
                image_gray = image.convert('L')
                image_gray_resized = image_gray.resize((image_size, image_size))
                image_array = asarray(image_gray_resized)

                data_array.append(image_array)
                label_class.append(label_track)

            label_track += 1

        self.img_data['data_array'] = data_array
        self.img_data['label'] = label_class
        self.img_data['labels'] = labels

        return self.img_data

    def separate(self):

        # separates image data and class label
        X = self.img_data['data_array']
        y = self.img_data['label']
        X = np.array(X)
        y = np.array(y)

        X = X.reshape((X.shape[0], -1))
        # print(X.shape)
        # print(y.shape)

        return X, y