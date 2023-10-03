import torch
import os
import pandas as pd
from PIL import Image

class VOCDataset(torch.utils.data.Dataset):
    '''
    Creates a Pytorch Dataset containing the data for training the model.
    '''
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S # naming from paper -> split_size
        self.B = B # number of bounding boxes
        self.C = C # number of classes

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        '''
        Get data in the format the model expects. 
        '''
        # Get labels from files which are specified in annoations csv
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x) # Get class labels as int, the rest as float
                    for x in label.replace("\n", "").split()
                ]
                boxes.append([class_label, x, y, width, height])

        # Get images 
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        # Do transformations in faster Pytorch tensors 
        boxes = torch.tensor(boxes)
        if self.transform: # For data augmentation. Change the bboxes, too.
            image, boxes = self.transform(image, boxes) 

        label_matrix = torch.zeros((self.S, self.S, self.C + 5*self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            # Calculate cell index 
            # Note: Tricky since the labels are done relative to the whole image and not to the cells. That's why this conversion is necessary.
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            # Calculate width and height
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # Label and keep track of the cells that contain an object 
            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return image, label_matrix

