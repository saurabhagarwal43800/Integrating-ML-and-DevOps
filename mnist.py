from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop, Adam, SGD
from keras.backend import clear_session
import  random

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


while accuracy < 90 :
    clear_session()
    counter=random.randint(1,5)
    learning_rate=random.choice([0.01,0.001,0.0001])
    epoch=random.randint(5,15)
    opt=[RMSprop,SGD,Adam]
    model = Sequential()
    for i in range(counter) :
            model.add(Dense(units=random.choice([512,128,256]),activation="relu",input_shape=(784,)))
      
    counter = counter +1
    print("counter is ",counter)
    
    model.add(Dense(units=10,activation="softmax"))      
    learning_rate = random.choice([0.01,0.001,0.0001]) #learning_rate/10
    print("learning_rate is", learning_rate)
    opt=random.choice(opt)
    model.compile(optimizer=(opt)(learning_rate),loss="categorical_crossentropy",metrics=["accuracy"])
    model.fit(X_train,y_train,batch_size=32,epochs=epoch,verbose=1)
    model.summary()
    Accuracy = model.evaluate(x=X_test,y=y_test,batch_size=32)
    print("Accuracy: ",(Accuracy[1]*100))
    accuracy = Accuracy[1]*100
    print(accuracy)
  