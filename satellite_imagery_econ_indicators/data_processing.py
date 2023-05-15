import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split

#######################################################
# LOADING AND PREPROCESSING DATA
#######################################################

# Loading image features and normalizing values
class data_bundle:
    def __init__(self, images_path, popn_path, labels_path, label_type, augmented):
        self.images = np.load(images_path)
        self.labels = np.load(labels_path)
        self.label_type = label_type
        self.image_height = self.images[0].shape[0]
        self.image_width = self.images[0].shape[1]
        self.image_channels = self.images[0].shape[2]

        if augmented:
            self.popn = np.load(popn_path)
        else:
            self.popn = pd.read_csv(popn_path)[["2019_figure_est"]]

    def normalize_images(self):
        features_img = self.images.astype('float32')
        features_img = features_img/255.0
        self.images = features_img

    def normalize_popn(self):
        features_popn = self.popn
        features_popn = (features_popn - 50000)/450000 
        self.popn = features_popn

    def bundle_inputs(self):
        features = np.array(list(zip(self.images, np.array(self.popn))))
        self.features = features

    def split_data(self, test_size_1 = 0.4, test_size_2 = 0.5, random_state = 1945):
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=test_size_1, random_state=random_state, stratify=self.labels)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=test_size_2, random_state=random_state, stratify=y_test)

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

    def augment_training_data(self):
        combined = []

        for i in range(self.X_train.shape[0]):
            combined.append([self.X_train[i][0], self.X_train[i][1], self.y_train[i]])

        augmented = []

        time = 1

        for i in combined:

            if i[2] != 1:
                augmented.append(i)

                if time == 1:
                    augmented.append(np.array([np.rot90(i[0]), i[1], i[2]]))
                    time = 2

                elif time == 2:
                    augmented.append(np.array([np.rot90(np.rot90(i[0])), i[1], i[2]]))
                    time = 3

                elif time == 3:
                    augmented.append(np.array([np.rot90(np.rot90(np.rot90(i[0]))), i[1], i[2]]))
                    time = 1
            
            elif i[2] == 1:
                augmented.append(i)
                augmented.append(np.array([np.rot90(i[0]), i[1], i[2]]))
                augmented.append(np.array([np.rot90(np.rot90(i[0])), i[1], i[2]]))
                augmented.append(np.array([np.rot90(np.rot90(np.rot90(i[0]))), i[1], i[2]]))
                augmented.append(np.array([np.fliplr(i[0]), i[1], i[2]]))
                augmented.append(np.array([np.flipud(i[0]), i[1], i[2]]))

            elif i[2] == 2:
                augmented.append(i)
                augmented.append(np.array([np.flipud(i[0]), i[1], i[2]]))

        np.random.shuffle(np.array(augmented))
        self.X_train = augmented[:,:2]
        self.y_train = augmented[:,2]

    def unbundle_and_shape_inputs(self):
        X_train_img = self.X_train[:,0]
        X_val_img = self.X_val[:,0]
        X_test_img = self.X_test[:,0]

        X_train_popn = self.X_train[:,1]
        X_val_popn = self.X_val[:,1]
        X_test_popn = self.X_test[:,1]

        # Image inputs
        X_tri = np.zeros((len(self.X_train),self.image_height,self.image_width,self.image_channels))
        for i, element in enumerate(X_train_img):
            X_tri[i] = element
        self.X_tri = X_tri

        X_vai = np.zeros((len(self.X_val),self.image_height,self.image_width,self.image_channels))
        for i, element in enumerate(X_val_img):
            X_vai[i] = element
        self.X_vai = X_vai

        X_tei = np.zeros((len(self.X_test),self.image_height,self.image_width,self.image_channels))
        for i, element in enumerate(X_test_img):
            X_tei[i] = element
        self.X_tei = X_tei

        #Population figure inputs
        X_trp = np.zeros((len(self.X_train),))
        for i, element in enumerate(X_train_popn):
            X_trp[i] = element
        self.X_trp = X_trp

        X_vap = np.zeros((len(self.X_val),))
        for i, element in enumerate(X_val_popn):
            X_vap[i] = element
        self.X_vap = X_vap

        X_tep = np.zeros((len(self.X_test),))
        for i, element in enumerate(X_test_popn):
            X_tep[i] = element
        self.X_tep = X_tep

    def onehot_encode_labels(self):
        num_classes = 4
        y_train_onehot = keras.utils.to_categorical((self.y_train-1), num_classes=num_classes)
        y_val_onehot = keras.utils.to_categorical((self.y_val-1), num_classes=num_classes)
        y_test_onehot = keras.utils.to_categorical((self.y_test-1), num_classes=num_classes)

        self.y_train_onehot = y_train_onehot
        self.y_val_onehot = y_val_onehot
        self.y_test_onehot = y_test_onehot


# The resulting input sets are: self.X_tri, self.X_vai, self.X_tei (Images) and self.X_trp, self.X_vap, self.X_tep (Population)
# The resulting tragets are: self.y_train_onehot, self.y_val_onehot and self.y_test_onehot