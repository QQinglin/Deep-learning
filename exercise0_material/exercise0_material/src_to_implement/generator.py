from cProfile import label
from itertools import batched
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from fontTools.t1Lib import decryptType1
from skimage.transform import resize

class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size,
                 rotation=False, mirroring=False, shuffle=False):
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self. mirroring = mirroring
        self.shuffle = shuffle

        self.epoch = 0
        self.current_index = 0
        self.image_file = []
        self.data = []

        """
        labels = {
            "image_001": 3,
            "image_002": 7,
            "image_003": 9,
            "image_004": 5 
        """
        ##
        with open(label_path,"r") as f:
            self.labels = json.load(f)

        all_files = os.listdir(self.file_path) # all_files is a list containing all file name
        for file in all_files:
            if file.endswith(".npy"):
                self.image_file.append(file.replace(".npy", ""))  # self.image_file -> image_004.npy

        if self.shuffle:
            np.random.shuffle(self.image_file)

        self.class_mapping = {
            0: "airplane",
            1: "automobile",
            2: "bird",
            3: "cat",
            4: "deer",
            5: "dog",
            6: "frog",
            7: "horse",
            8: "ship",
            9: "truck"
        }

    def _load_image(self,filename): # filename -> image_004.npy
        img_path = os.path.join(self.file_path,filename + '.npy')
        img = np.load(img_path)
        img_reseized = resize(img, self.image_size,anti_aliasing = True)
        return  img_reseized

    def _apply_augmentation(self,image):
        if self.mirroring and np.random.rand() > 0.5:
            image = np.fliplr(image)
        if self.rotation:
            k = np.random.choice([0, 1, 2, 3])
            image = np.rot90(image,k)
        return image

    def next(self):
        start = self.current_index
        end = start + self.batch_size
        if end >= len(self.image_file):
            batch_files = self.image_file[start: ] + self.image_file[: end - len(self.image_file)]
        else:
            batch_files = self.image_file[start:end]  # batch_files -> image_004.npy


        images = []
        labels = []

        for file in batch_files:
            img = self._load_image(file) # file -> image_004.npy
            img = self._apply_augmentation(img)
            images.append(img)
            labels.append(self.labels[file])

        self.current_index = self.current_index + self.batch_size

        if self.current_index > len(self.image_file):
            self.current_index = self.current_index % len(self.image_file)
            self.epoch = self.epoch + 1
            if self.shuffle:
                np.random.shuffle(self.image_file)

        if self.current_index == len(self.image_file):
            if self.shuffle:
                np.random.shuffle(self.image_file)

        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)

        return  images, labels

    def current_epoch(self):
        return self.epoch

    def class_name(self, label):
        return self.class_mapping.get(label, "Unknown")

    def show(self):
        images, labels = self.next()
        fig, axes = plt.subplots(1,len(images),figsize = (15,5))
        if len(images) == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ax.imshow(images[i])
            ax.set_title(self.class_name(labels[i]))
            ax.axis("off")

        plt.show()