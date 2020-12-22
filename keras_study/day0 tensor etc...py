import numpy as np
'''x = np.array([[[12,3,4,5,6],[12,3,4,4,5],[1,23,4,5,6]],[[12,3,4,5,6],[12,3,4,4,5],[1,24,53,3,3]]])
print(x.ndim) #x tensor의 축갯수
#tensor의 축 배열이 행끼리 축이 일치하지 않으면 diffrent lengths oqshape에러
#shape 각 축을 따라 얼마나 많은 차워이 있는지 나타내는 tuple x는 (2,3,5)
#데이터 타입: tenso에 포함된 데이터 타입 float32,unit8,float64 값 하나에 32byte라는 뜻 등 char도 있다
#tenso는 사전에 할당되어서 연속된 메모리에 저장되어야해서
#numpy배열은 가변길이의 문자열 지원 안함'''

from keras.datasets import mnist
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()
print(train_images.ndim)
print(train_images.shape)
print(train_images.dtype)
digit = train_images[4]
import matplotlib.pyplot as plt
plt.imshow(digit,cmap = plt.cm.binary)

myslice = train_images[:,7:-14,7:-14] #tensor 슬라이싱은 이런식으로
print(myslice.shape)

#딥러닝모델은 한 번에 모든 데이터셋을 처리하지 않고 작은 배치로 나눔
n=0
batch = train_images[128*n:128*(n+1)]#n번째 batch


#텐서 연산
from keras import layers
layers.Dense(512,activation='relu')
#== output = relu(dot(w, input) + b) #dot:점곱 스칼라곱 내적


