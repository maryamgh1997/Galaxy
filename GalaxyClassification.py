import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


dataset = pd.read_csv('GalaxyZoo1_DR_table2.csv')
dataset.head()
dataset.info()
dataset.describe().T
print(f"dataset shape:\t {dataset.shape}")


#Drop Duplicate Values
dataset.drop_duplicates(inplace=True)
dataset.reset_index(drop=True, inplace=True)
dataset.shape
print(f"dataset shape:\t {dataset.shape}")

#Finding Missing Values
features_na = [features for features in dataset.columns if dataset[features].isnull().sum() > 0]
if(len(features_na)>0):
    for feature in features_na:
        print("{}: {} %".format(feature, np.round(dataset[feature].isnull().mean()*100, 4)))
else:
    print("No any missing value found")
    

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13:16].values
print(f"X shape: {X.shape}\ny shape: {y.shape}")

#to split into train and validation and test set
from sklearn.model_selection import train_test_split
X_train, x_v, y_train, y_v = train_test_split(X, y, test_size = 0.3, random_state = 0)
X_val, X_test, y_val, y_test = train_test_split(x_v,y_v,test_size = 0.15, random_state = 0)


#standard data
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
X_train = standardScaler.fit_transform(X_train)
X_test = standardScaler.transform(X_test)
X_val = standardScaler.transform(X_val)

# Importing libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2


#creating our model
model = Sequential()
model.add(Dense(32, input_dim=10, activation='relu'))
model.add(Dense(64, activation='relu' , kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(3, activation='sigmoid'))

#to plot our model and save it
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True)

#Early stopping function
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(patience= 5, monitor='val_accuracy', mode='max',  verbose=1, restore_best_weights=True)

#defining scheduled learning rate
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import RMSprop
lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.9)
opt = RMSprop(learning_rate=lr_schedule)

#to compile our model
model.compile(loss='binary_crossentropy', optimizer=opt , metrics = ['accuracy'])

# Fitting model
history = model.fit(X_train, y_train, batch_size = 64, epochs = 20, validation_data=(X_val, y_val), callbacks=[ es])

#to evaluate our model on test set
model.evaluate(X_test , y_test)

#to plot
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()