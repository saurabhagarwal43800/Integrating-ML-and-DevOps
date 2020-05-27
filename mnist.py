from keras.datasets import mnist

dataset=mnist.load_data('mymnist.db')
train,test=dataset
X_train,y_train=train
X_test,y_test=test
img1=X_train[0]
img1_label=y_train[0]

import matplotlib.pyplot as plt
plt.imshow(img1,cmap='gray')
img1_1d=img1.reshape(28*28)

X_train_1d=X_train.reshape(-1,28*28)

X_train=X_train_1d.astype('float32')

from keras.utils.np_utils import to_categorical

y_train_cat=to_categorical(y_train)

from keras.models import Sequential

from keras.layers import Dense

model=Sequential()

model.add(Dense(units=512,input_dim=28*28,activation='relu'))

model.add(Dense(units=256,activation='relu'))

model.add(Dense(units=128,activation='relu'))

model.add(Dense(units=32,activation='relu'))

model.add(Dense(units=10,activation='softmax'))

model.summary()

from keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(),loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train_cat,epochs=3)

accuracy=model.evaluate(x=X_test, y=y_test, batch_size=32)
print("Accuracy",accuracy[1])
Accuracy=accuracy[1]
print(Accuracy)

#plt.imshow(X_test[9999])

#y_test[9999]

#test_img=X_test[9999].reshape(-1,28*28)

#test_img=test_img.astype('float32')

#model.predict(test_img)
