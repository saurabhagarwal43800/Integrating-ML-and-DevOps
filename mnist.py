from keras.datasets import mnist

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import RMSprop

dataset=mnist.load_data()

train,test=dataset
X_train,y_train=train

X_test,y_test=test



X_train=X_train.reshape(-1,28*28)

X_train=X_train.astype('float32')

y_train=to_categorical(y_train)




X_test=X_test.reshape(-1,28*28)

X_test=X_test.astype('float32')
y_test=to_categorical(y_test)



accuracy=0
counter=1
learning_rate=0.1
epoch=10

while accuracy < .90 :
    model = Sequential()
    for i in range(counter) :
            model.add(Dense(units=128,activation="relu",input_shape=(784,)))
      
    counter = counter +1
    print("counter is ",counter)
    
    model.add(Dense(units=10,activation="softmax"))      
    learning_rate = learning_rate/10
    print("learning_rate is", learning_rate)
    model.compile(optimizer=RMSprop(learning_rate),loss="categorical_crossentropy",metrics=["accuracy"])
    model.fit(X_train,y_train,batch_size=32,epochs=epoch,verbose=1)
    model.summary()
    Accuracy = model.evaluate(x=X_test,y=y_test,batch_size=32)
    print("Accuracy: ",int(Accuracy[1])*100)
    accuracy = int(Accuracy[1])*100
    print(accuracy)