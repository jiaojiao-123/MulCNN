# 用数据训练模型,去掉了最后一列的nan
# 保存pca模型，保存cnn模型
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

# 标签类型数量
label_num = 4
# 把每条数据转换成方阵长度
sequence_length = 185
# 降维数
pca_num = 240 #

# 模型相关参数
epoch = 300 # 300
batch = 32
my_seed = 26  #26
np.random.seed(my_seed)
random.seed(my_seed)
tf.random.set_seed(my_seed)
# tf.set_random_seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)


# 数据集选择
train_name = []
train_name.append('/home/aita/4444/jiao/cellclassification/mydata/Xin/train_small.csv')  # Segerstolpe,Lawlor,Baron,Xin
test_name = []
test_name.append('/home/aita/4444/jiao/cellclassification/mydata/Xin/test_small.csv')
valid_name = []
valid_name.append('/home/aita/4444/jiao/cellclassification/mydata/Xin/valid_small.csv')
bridge_name = []
bridge_name.append('/home/aita/4444/jiao/cellclassification/mydata/Xin/bridge.csv')
model_name = []
model_name.append('/home/aita/4444/jiao/cellclassification/modelsave/cnn_Xin3.h5')
pca_model_name = []
pca_model_name.append('/home/aita/4444/jiao/cellclassification/modelsave/pca_Xin3.m')

# 处理训练集,提取特征&标签
def traindata(data_name = train_name):
    label_list = []
    feature_list = []
    feature_sequence = []
    bridge = []
    for num in range(len(data_name)):
        print("train data_name",data_name)
        # 读数据
        expression = pd.read_csv(data_name[0], header=None)
        # 提取标签
        label = []
        label = expression[1]
        label = np.array(label)
        label = label[1:]
        # print('输入标签',label)
        # 把标签转为数字
        bridge = pd.read_csv(bridge_name[0],header=None)
        bridge = np.array(bridge)
        temp = []
        for i in range(len(label)):
            for j in range(bridge.shape[1]):
                if label[i] == bridge[0][j]:
                    temp.append(bridge[1][j])
        label = np.array(temp)
        # print('调整标签格式',label)
        # 换成onehot编码,从左往右第几个是1就是第几类
        label = to_categorical(label, num_classes= label_num+1)
        label = np.delete(label, 0, axis=1)
        # print('调整标签编码',label)
        for i in range(len(label)):
            temp = []
            for j in range(len(label[i])):
                temp.append(int(label[i][j]))
            label_list.append(temp)
        label_list = np.array(label_list)

        # 提取特征（每行是一个cell
        expression = np.array(expression)
        expression = expression[1:,2:expression.shape[1]-1]
        expression = np.array(expression)
        for i in range(len(expression)):
            temp = []
            for j in range(len(expression[i])):
                temp.append(float(expression[i][j]))
            feature_list.append(temp)
        feature_list = np.array(feature_list)
        # CPM标准化特征
        for i in range(feature_list.shape[0]):
            for j in range(feature_list.shape[1]):
                if feature_list[i][j] != 0:
                    feature_list[i][j] = np.log2((feature_list[i][j]*1000000)/feature_list.shape[1])
        print('CPM标准化后',feature_list.shape)

        # 把每条特征变成矩阵
        for i in range(1, feature_list.shape[0] + 1):
            temp = list(feature_list[i - 1])
            temp = temp + [0] * ((sequence_length * sequence_length) - len(feature_list[i - 1]))
            temp = np.array(temp)
            temp = temp.reshape((sequence_length, sequence_length, 1))
            feature_sequence.append(temp)
        feature_sequence = np.array(feature_sequence)

        return feature_sequence,label_list,feature_list
train = traindata()
print('train[0]方特征',train[0].shape) #a[0]特征
print('train[1]标签',train[1].shape) #a[1]标签
print('train[2]条特征',train[2].shape)


# PCA降维
pca = PCA(n_components= pca_num)  # n_components保留的主成分个数，='mle'时自动选取特征数，或者自定义=1222
pca.fit(train[2])  # 训练PCA模型
train_pca = pca.transform(train[2])
joblib.dump(pca,pca_model_name[0])  #保存pca模型
print('降维后train_pca', train_pca.shape)


# 模型
def cnnmodel():
    inputA = Input(shape = (sequence_length,sequence_length,1))
    inputB = Input(shape= (pca_num,1))
    # 卷积层
    x1 = layers.Conv2D(filters= 256, kernel_size=(2, 2), activation='relu',input_shape=(sequence_length, sequence_length, 1))(inputA)
    x1 = layers.MaxPooling2D(2, 2)(x1)
    x1 = Dropout(0.25)(x1)

    x2 = layers.Conv2D(filters= 128, kernel_size=(5, 5), activation='relu',input_shape=(sequence_length, sequence_length, 1))(inputA)
    x2 = layers.MaxPooling2D(2, 2)(x2)
    x2 = Dropout(0.25)(x2)

    x4 = layers.MaxPooling2D(3, 3)(inputA)

    # PCA
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

    # 全连接层
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
# 输出模型各层的参数状况
# model.summary()

# 处理验证集
def validdata(data_name = valid_name):
    label_list = []
    label_list_true = []
    feature_list = []
    feature_sequence = []
    bridge = []
    for num in range(len(data_name)):
        print("test data_name",data_name)
        # 读数据
        expression = pd.read_csv(data_name[0], header=None)
        # 提取标签
        label = []
        label = expression[1]
        label = np.array(label)
        label = label[1:]
        # print('输入标签',label)
        # 把标签转为数字
        bridge = pd.read_csv(bridge_name[0],header=None)
        bridge = np.array(bridge)
        temp = []
        for i in range(len(label)):
            for j in range(bridge.shape[1]):
                if label[i] == bridge[0][j]:
                    temp.append(bridge[1][j])
        label = np.array(temp)
        label_list_true = label
        # print('调整标签格式',label)
        # 换成onehot编码,从左往右第几个是1就是第几类
        label = to_categorical(label, num_classes= label_num+1)
        label = np.delete(label, 0, axis=1)
        # print('调整标签编码',label)
        for i in range(len(label)):
            temp = []
            for j in range(len(label[i])):
                temp.append(int(label[i][j]))
            label_list.append(temp)
        label_list = np.array(label_list)
        # print('标签',label_list)
        # print('标签',label_list.shape)

        # 提取特征（每行是一个cell
        expression = np.array(expression)
        expression = expression[1:,2:expression.shape[1]-1]
        expression = np.array(expression)
        for i in range(len(expression)):
            temp = []
            for j in range(len(expression[i])):
                temp.append(float(expression[i][j]))
            feature_list.append(temp)
        feature_list = np.array(feature_list)
        # CPM标准化特征
        for i in range(feature_list.shape[0]):
            for j in range(feature_list.shape[1]):
                if feature_list[i][j] != 0:
                    feature_list[i][j] = np.log2((feature_list[i][j]*1000000)/feature_list.shape[1])
        # print('CPM标准化后',feature_list.shape)

        # 把每条特征变成矩阵
        for i in range(1, feature_list.shape[0] + 1):
            temp = list(feature_list[i - 1])
            temp = temp + [0] * ((sequence_length * sequence_length) - len(feature_list[i - 1]))
            temp = np.array(temp)
            temp = temp.reshape((sequence_length, sequence_length, 1))
            feature_sequence.append(temp)
        feature_sequence = np.array(feature_sequence)

        return feature_sequence,label_list,label_list_true,feature_list
valid = validdata()
print('valid[0]方特征',valid[0].shape) #a[0]特征
print('valid[1]标签',valid[1].shape) #a[1]标签
# print('test[2]标签数字格式',test[2])
print('valid[3]长特征',valid[3].shape)


# 载入pca模型
# pca2 = joblib.load(pca_model_name[0])
# valid_pca = pca2.transform(valid[3])
# print('降维后valid_pca', valid_pca.shape)
valid_pca = pca.transform(valid[3])

# 预测
predict = model.predict([valid[0],valid_pca])
# # 评估操作
validloss, validacc = model.evaluate([valid[0],valid_pca], valid[1], verbose=2)
print('valid loss:',validloss)
print('valid acc:',validacc)
print('predict',predict[0])

# 把预测出的结果变成数字
predict_max = []
for i in range(predict.shape[0]):
    temp = []
    temp = np.argmax(predict[i])
    temp = temp +1
    predict_max.append(temp)
predict_max = np.array(predict_max)
# print('predict_max',predict_max)

data_true = []
temp_data = valid[2]
for i in range(len(temp_data)):
    data_true.append(int(temp_data[i]))
data_true = np.array(data_true)
data_pred = predict_max
# print('data_true',data_true)
# print('data_pred',data_pred)


# 计算各个评价指标的值
f1 = f1_score( data_true, data_pred, average='macro' )   #越1越好
precision = precision_score(data_true, data_pred, average='macro') #越1
recall = recall_score(data_true, data_pred, average='macro') #越1
accuracy = accuracy_score(data_true, data_pred)
print('accuracy,f1, precision, recall 结果为：',accuracy,f1, precision, recall)

# 计算ARI
ari = metrics.adjusted_rand_score(data_true,data_pred)
print('ari',ari)


# 保存模型
model.save(model_name[0])


