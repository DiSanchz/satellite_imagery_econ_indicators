import keras
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.layers import Dense, Flatten, Input, Concatenate, Dropout
from keras.regularizers import l2
from keras.applications.vgg16 import VGG16
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

#######################################################
# VGG16 model
#######################################################

class vgg16_seei:

    def __init__(self, input_shape, dropout_rate, learning_rate):

        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.dual = False
    
    def load_and_freeze_vgg16(self):

        # Import VGG16
        self.model = VGG16(include_top=False, input_shape=self.input_shape)

        # Freezing VGG16 layers
        for layer in self.model.layers:
            layer.trainable = False

    def enable_single_input(self):

        # Flattening VGG16 output
        self.inputs = layers.Flatten()(self.model.layers[-1].output)


    def enable_dual_input(self):

        # Entry layer for instance population inputs
        integer_input = Input(shape=(1,), name='integer_input')
        self.integer_input = integer_input

        # Flattening VGG16 output
        image_input = layers.Flatten()(self.model.layers[-1].output)

        # Merge VGG16 output with instance population
        self.inputs = Concatenate()([image_input, integer_input])

        self.dual = True

    def add_dense_layers(self):

        # Dense layers
        dense_1 = layers.Dense(256, activation='relu')(self.inputs)
        dropout_1 = Dropout(self.dropout_rate)(dense_1)  

        dense_2 = layers.Dense(128, activation='relu')(dropout_1)
        dropout_2 = Dropout(self.dropout_rate)(dense_2)  

        dense_3 = layers.Dense(64, activation='relu')(dropout_2)
        dropout_3 = Dropout(self.dropout_rate)(dense_3)  

        dense_4 = layers.Dense(32, activation='relu')(dropout_3)
        dropout_4 = Dropout(self.dropout_rate)(dense_4)  

        dense_f = layers.Dense(16, activation='relu')(dropout_4)

        # Output layer 
        self.output = layers.Dense(4, activation='softmax')(dense_f)

    def define_and_compile_model(self, optimizer='adam'):

        if self.dual:
        
            self.defined_model = Model(inputs=[self.model.inputs, self.integer_input], outputs=self.output)

        else:

            self.defined_model = Model(inputs=self.model.inputs, outputs=self.output)

        self.defined_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    
    def fit_model(self, batch_size = 64, epochs = 100, train_in = None, train_tar = None, 
                  validation_in = None, validation_tar = None ,es_patience = 25):

        # Early stopping 
        es = EarlyStopping(monitor='val_accuracy', mode='max', patience=25,  restore_best_weights=True)
        
        # Model fit
        self.fit = self.defined_model.fit(train_in, train_tar, validation_data=(validation_in, validation_tar), 
                                          batch_size=batch_size ,epochs=epochs, verbose=1, callbacks=es)
        
    
    def predict(self, test_in = None):
            
            self.predictions = self.defined_model.predict(test_in)

    
    def save(self, path = None):
        
        self.defined_model.save(path)


    def load(self, path = None):
        
        self.defined_model = keras.models.load_model(path)