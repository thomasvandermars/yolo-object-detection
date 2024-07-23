import numpy as np
from keras.utils import Sequence
from .pre_processing_np import preprocess_np, preprocess_with_augmentation_np

class data_generator(Sequence):
    
    def __init__(self, images, labels, batch_size, params, augment_params):
        
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.params = params
        self.augment_params = augment_params
        pass
    
    def __len__(self):
        
        return int(np.ceil(len(self.labels) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        
        # extract the batch
        batch_x = self.images[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        # preprocess image, label, and true boxes
        X, Y = [], []
        
        # if no augmentation parameters are passed in, we preprocess w/o augmentation
        if self.augment_params == None:
            for i in range(len(batch_y)):
                prep_img, prep_lbl = preprocess_np(image = batch_x[i], 
                                                   label = batch_y[i], 
                                                   params = self.params)
                X.append(prep_img)
                Y.append(prep_lbl)
                pass
            pass
        else: # if augmentation parameters are defined, we preprocess with augmentation
            for i in range(len(batch_y)):
                prep_img, prep_lbl = preprocess_with_augmentation_np(image = batch_x[i],
                                                                     label = batch_y[i], 
                                                                     params = self.params,
                                                                     augment_params = self.augment_params)
                X.append(prep_img)
                Y.append(prep_lbl)
                pass
            pass
        

        return np.array(X), np.array(Y)