import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import *
from sklearn.preprocessing import *
from keras import optimizers
import matplotlib.pyplot as plt
data_1 = pd.read_csv('MLHW/train-v3.csv')
data_2 = pd.read_csv('MLHW/valid-v3.csv')
data_3 = pd.read_csv('MLHW/test-v3.csv')
data_title=['bedrooms','bathrooms','sqft_living','sqft_lot','condition','grade','sqft_above','sqft_basement','zipcode','lat','long','sqft_living15','sqft_lot15','floors','waterfront','view','sale_yr','sale_month','sale_day']
data_1.drop(['id'],inplace=True,axis=1)
data_2.drop(['id'],inplace=True,axis=1)
data_3.drop(['id'],inplace=True,axis=1)

Y_train = data_1["price"].values
X_train = data_1[data_title].values
Y_valid = data_2["price"].values
X_valid = data_2[data_title].values
X_test = data_3[data_title].values

X_train=scale(X_train)
X_valid=scale(X_valid)
X_test=scale(X_test)
model = Sequential()
model.add(Dense(40, input_dim=X_train.shape[1],  kernel_initializer='normal',activation='relu'))
model.add(Dense(32,  kernel_initializer='normal',activation='relu'))
model.add(Dense(32,  kernel_initializer='normal',activation='relu'))
model.add(Dense(40,  kernel_initializer='normal',activation='relu'))
model.add(Dense(40,  kernel_initializer='normal',activation='relu'))


model.add(Dense(1,  kernel_initializer='normal'))
model.compile(loss='MAE', optimizer='adam')
nb_epoch = 199
batch_size = 64

fn=str(nb_epoch)+'_'+str(batch_size)
TB=TensorBoard(log_dir='logs/'+fn, histogram_freq=0)
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1,validation_data=(X_valid, Y_valid),callbacks=[TB])
model.save('MLHW/'+fn+'.h5')
Y_predict = model.predict(X_test)
np.savetxt('MLHW/test.csv', Y_predict, delimiter = ',')
