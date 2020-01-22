import scipy.io as sio
import numpy as np
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten,Input
from keras.layers import BatchNormalization
from sklearn.model_selection import cross_val_predict
from keras.layers import Conv1D,Conv2D,Conv3D, MaxPooling2D, MaxPooling1D, Reshape, GlobalAveragePooling1D
from keras.layers import SeparableConv2D, DepthwiseConv2D, AveragePooling2D
from keras.constraints import max_norm
from keras.callbacks import ModelCheckpoint
import pip
import hdf5storage

# Set random seed
np.random.seed(0)
lr = .001


## Veri setinin yüklenmesi ve düzenlenmesi
data_train = hdf5storage.loadmat('HamVeri.mat')['HamVeri']
data_label = sio.loadmat('Valence.mat')['Valence']

#data_test = sio.loadmat('Valence.mat')['Valence']

data_train = data_train.reshape(1125,3,14,7680)
data_label = data_label.transpose()

###############################################################################

## Sinyallerin evrişim işlemi için farklı boyutlarda tanımlanması
input1   = Input(shape = (3, 14, 7680))
Chans = 14
Samples = 7680
F1 = 4
kernLength = 32
dropoutType = 'Dropout'

block1= Conv2D(F1, (1, kernLength), padding = 'same',input_shape = (3, Chans, Samples),use_bias = False)(input1)
block1= BatchNormalization(axis = 1)(block1)
block1= DepthwiseConv2D((3, 3), use_bias = False, 
                                   depth_multiplier = 4,
                                   depthwise_constraint = max_norm(1.))(block1)
block1= BatchNormalization(axis = 1)(block1)
block1= Activation('elu')(block1)
block1= AveragePooling2D((1, 1))(block1)
block1= Dropout(0.25)(block1)
block2= SeparableConv2D(8, (1, 16),
                             use_bias = False, padding = 'same')(block1)
block2= BatchNormalization(axis = 1)(block2)
block2= Activation('elu')(block2)
block2= AveragePooling2D((1, 8))(block2)
block2= Dropout(0.25)(block2)
        
flatten= Flatten(name = 'flatten')(block2)
dense= Dense(1, name = 'dense', kernel_constraint = max_norm(0.25))(flatten)
sigmoid= Activation('sigmoid', name = 'sigmoid')(dense)
model =Model(inputs=input1, outputs=sigmoid)
model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

## Validation işlemi için veri setinin bölümlendirilmesi
X = np.array(data_train)
y = np.array(data_label)

## En iyi modelin kaydedilmesi
filepath = 'AV.weight.best.hdf5'
checkpointA = ModelCheckpoint(filepath,monitor = 'acc', verbose = 1, save_best_only = True, mode = 'max')
callback_listdene = [checkpointA]
from sklearn import cross_validation
cv = cross_validation.KFold(len(data_train), n_folds=10,shuffle=False) ## Eğitim verisinin bölümlendirilme işlemi
num = 10
avg_train=0
total_sum = 0

## Eğitim işlem,i adımı
for traincv, testcv in cv:
    avg_train=(model.fit(data_train[traincv],data_label[traincv],validation_data=(data_train[testcv],data_label[testcv]),batch_size=30, epochs=10,callbacks=callback_listdene,verbose = 0))
    total_sum = (np.array(avg_train.history['val_acc']) + total_sum) ## ortalama başarımın hesaplanması
    
    
    y_pred = cross_val_predict(model, X, y, cv=cv)
accuracy = accuracy_score(y_pred.astype(int), y.astype(int))

print(accuracy)
    
    
    
avg = total_sum/10
ort = avg
sum(ort)
print('the average is',top/10)
top = 0
