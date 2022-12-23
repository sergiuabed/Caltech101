from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys

#DATASET_PATH = "./101_ObjectCategories/"

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

        self.label_dict = dict()    # mapping from label in the form of string to integer

        self.imageLabelList = []
        self.label_index = dict()   # dict with labels as key and list of indexes of "imageLabelList" as value

        print('./Caltech101/' + split + ".txt")
        with open('./Caltech101/' + split + ".txt", 'r') as f:
            for line in f:
                _class = line.strip().split('/')[0]
                if _class == 'BACKGROUND_Google':
                    continue

                if _class not in self.label_dict.keys():
                    self.label_dict[_class] = len(self.label_dict.keys())

                if _class not in self.label_index.keys():
                    self.label_index[_class] = []

                imgpath = root + "/" + line.strip()

                self.label_index[_class].append(len(self.imageLabelList))
                self.imageLabelList.append((imgpath, _class))


    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        if isinstance(index, slice):
            stop = index.stop if index.stop is not None else len(self.imageLabelList)
            start = index.start if index.start != None else 0
            step = index.step if index.step != None else 1

            if self.transform is None:
                return [(pil_loader(self.imageLabelList[i][0]), self.label_dict[self.imageLabelList[i][1]]) for i in range(start, stop, step)]
            else:
                return [(self.transform(pil_loader(self.imageLabelList[i][0])), self.label_dict[self.imageLabelList[i][1]]) for i in range(start, stop, step)]

            

        image, label = self.imageLabelList[index]
        image = pil_loader(image)
        label = self.label_dict[label]

        #image, label = 1, 1 # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label
        #return image

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.imageLabelList) # Provide a way to get the length (number of elements) of the dataset
        return length

    def partition(self, train_part):
        assert train_part <= 1
        
        train_set_idxs = []
        val_set_idxs = []

        for c in self.label_index:
            div = int(train_part*len(self.label_index[c]))

            idxs_train = self.label_index[c][:div]
            idxs_val = self.label_index[c][div:]

            train_set_idxs.extend(idxs_train)
            val_set_idxs.extend(idxs_val)

        return train_set_idxs, val_set_idxs

#            t = [self.imageLabelList[i] for i in idxs_train]
#            v = [self.imageLabelList[i] for i in idxs_val]
#
#            train_set.extend(t)
#            val_set.extend(v)
