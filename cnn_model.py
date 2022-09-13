import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute,Input, Concatenate, Conv2D, Add, Activation, Lambda
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score
from numpy.random import seed
import random

label_num = 4
sequence_length = 185
pca_num = 240 
epoch = 300 
batch = 32
my_seed = 26  
np.random.seed(my_seed)
random.seed(my_seed)
tf.random.set_seed(my_seed)

def traindata(data_name = train_name):
    label_list = []
    feature_list = []
    feature_sequence = []
    bridge = []
    for num in range(len(data_name)):
        expression = pd.read_csv(data_name[0], header=None)
        label = []
        label = expression[1]
        label = np.array(label)
        label = label[1:]
        bridge = pd.read_csv(bridge_name[0],header=None)
        bridge = np.array(bridge)
        temp = []
        for i in range(len(label)):
            for j in range(bridge.shape[1]):
                if label[i] == bridge[0][j]:
                    temp.append(bridge[1][j])
        label = np.array(temp)
        label = to_categorical(label, num_classes= label_num+1)
        label = np.delete(label, 0, axis=1)
        for i in range(len(label)):
            temp = []
            for j in range(len(label[i])):
                temp.append(int(label[i][j]))
            label_list.append(temp)
        label_list = np.array(label_list)

        expression = np.array(expression)
        expression = expression[1:,2:expression.shape[1]-1]
        expression = np.array(expression)
        for i in range(len(expression)):
            temp = []
            for j in range(len(expression[i])):
                temp.append(float(expression[i][j]))
            feature_list.append(temp)
        feature_list = np.array(feature_list)

        for i in range(feature_list.shape[0]):
            for j in range(feature_list.shape[1]):
                if feature_list[i][j] != 0:
                    feature_list[i][j] = np.log2((feature_list[i][j]*1000000)/feature_list.shape[1])

        for i in range(1, feature_list.shape[0] + 1):
            temp = list(feature_list[i - 1])
            temp = temp + [0] * ((sequence_length * sequence_length) - len(feature_list[i - 1]))
            temp = np.array(temp)
            temp = temp.reshape((sequence_length, sequence_length, 1))
            feature_sequence.append(temp)
        feature_sequence = np.array(feature_sequence)

        return feature_sequence,label_list,feature_list
train = traindata()


pca = PCA(n_components= pca_num) 
pca.fit(train[2])  
train_pca = pca.transform(train[2])
joblib.dump(pca,pca_model_name[0])  

def cnnmodel():
    inputA = Input(shape = (sequence_length,sequence_length,1))
    inputB = Input(shape= (pca_num,1))
   
    x1 = layers.Conv2D(filters= 256, kernel_size=(2, 2), activation='relu',input_shape=(sequence_length, sequence_length, 1))(inputA)
    x1 = layers.MaxPooling2D(2, 2)(x1)
    x1 = Dropout(0.25)(x1)

    x2 = layers.Conv2D(filters= 128, kernel_size=(5, 5), activation='relu',input_shape=(sequence_length, sequence_length, 1))(inputA)
    x2 = layers.MaxPooling2D(2, 2)(x2)
    x2 = Dropout(0.25)(x2)

    x4 = layers.MaxPooling2D(3, 3)(inputA)

    x3 = Flatten()(inputB)
    x1 = Flatten()(x1)
    x2 = Flatten()(x2)
    x4 = Flatten()(x4)
    x = layers.concatenate([x1, x2])
    x = layers.concatenate([x, x3])
    x = layers.concatenate([x, x4])
    x = layers.BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = Dropout(0.25)(x)
    z = layers.Dense(label_num, activation='softmax')(x)
    model = Model(inputs= [inputA, inputB],outputs = z)
    optimizer = tf.keras.optimizers.SGD(lr=0.0001)
    model.compile(optimizer= optimizer, loss=tf.keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])
    return model

model = cnnmodel()
model.fit(x= [train[0],train_pca], y=train[1], batch_size=batch, epochs=epoch,shuffle= False)

def validdata(data_name = valid_name):
    label_list = []
    label_list_true = []
    feature_list = []
    feature_sequence = []
    bridge = []
    for num in range(len(data_name)):
        expression = pd.read_csv(data_name[0], header=None)
        label = []
        label = expression[1]
        label = np.array(label)
        label = label[1:]
        bridge = pd.read_csv(bridge_name[0],header=None)
        bridge = np.array(bridge)
        temp = []
        for i in range(len(label)):
            for j in range(bridge.shape[1]):
                if label[i] == bridge[0][j]:
                    temp.append(bridge[1][j])
        label = np.array(temp)
        label_list_true = label
        label = to_categorical(label, num_classes= label_num+1)
        label = np.delete(label, 0, axis=1)
        for i in range(len(label)):
            temp = []
            for j in range(len(label[i])):
                temp.append(int(label[i][j]))
            label_list.append(temp)
        label_list = np.array(label_list)

        expression = np.array(expression)
        expression = expression[1:,2:expression.shape[1]-1]
        expression = np.array(expression)
        for i in range(len(expression)):
            temp = []
            for j in range(len(expression[i])):
                temp.append(float(expression[i][j]))
            feature_list.append(temp)
        feature_list = np.array(feature_list)
        for i in range(feature_list.shape[0]):
            for j in range(feature_list.shape[1]):
                if feature_list[i][j] != 0:
                    feature_list[i][j] = np.log2((feature_list[i][j]*1000000)/feature_list.shape[1])

        for i in range(1, feature_list.shape[0] + 1):
            temp = list(feature_list[i - 1])
            temp = temp + [0] * ((sequence_length * sequence_length) - len(feature_list[i - 1]))
            temp = np.array(temp)
            temp = temp.reshape((sequence_length, sequence_length, 1))
            feature_sequence.append(temp)
        feature_sequence = np.array(feature_sequence)

        return feature_sequence,label_list,label_list_true,feature_list
valid = validdata()

valid_pca = pca.transform(valid[3])

predict = model.predict([valid[0],valid_pca])

model.save(model_name[0])
