from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

''' for see
import matplotlib.pyplot as plt
digit = test_images[10]
plt.imshow(digit,cmap = plt.cm.binary)
plt.show()'''

(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

network = models.Sequential()
network.add(layers.Dense(512,activation = 'relu',input_shape = (28*28,)))
#activation = 활성화, relu = f(x) = max(0,x),shape = 28x28 pixel
network.add(layers.Dense(10,activation = 'softmax'))
#softmax : output 값을 확률값으로 바꿔주는 function
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics = ['accuracy'])
#이때 사용하는게 loss function -> 여기서는 label값과 softmax값의 loss 추출해주는 crossentropy
 
#0~255값인 unit8 배열에서 0~1값인 float32배열로 바꿈
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape(10000,28*28)
test_images = test_images.astype('float32')/255

#레이블을 범주형으로 인코딩 ?
train_labels= to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#학습 fit method
network.fit(train_images,train_labels,epochs=5,batch_size=128)
network.fit(train_images,train_labels,epochs=5,batch_size=128)

#check
test_loss,test_acc = network.evaluate(train_images,train_labels)

print('test_acc:',test_acc)
