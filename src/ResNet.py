# ResNet
import tensorflow as tf
import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.datasets import cifar10
from keras import losses
from keras import optimizers
from keras import layers
from keras.layers import Dense, AveragePooling2D, Flatten, MaxPool2D, Input, BatchNormalization, Activation, Conv2D
from keras.models import Model 
from keras.regularizers import l2
import os
from datetime import date

def conv2d_bn(x, filters, kernel_size, weight_decay=1e-4, strides=(1,1)):
    layer = Conv2D(filters, kernel_size, padding='same', strides=strides, kernel_regularizer=l2(weight_decay))(x)
    layer = BatchNormalization(axis=-1)(layer)
    return layer

def conv2d_bn_relu(x, filters, kernel_size, weight_decay=1e-4, strides=(1,1)):
    layer = Conv2D(filters, kernel_size, padding='same', strides=strides, kernel_regularizer=l2(weight_decay))(x)
    if layer is None:
        raise ValueError("Conv2D returned None. Check the input and parameters.")
    layer = BatchNormalization(axis=-1)(layer)
    if layer is None:
        raise ValueError("BatchNormalization returned None. Check the input and parameters.")
    layer = Activation('relu')(layer)
    if layer is None:
        raise ValueError("Activation returned None. Check the input and parameters.")
    return layer

def ResidualBlock(x,filters,kernel_size,weight_decay,downsample=True):
    if downsample:
        residual_x=conv2d_bn(x,filters,kernel_size=1,strides=2)
        stride=2
    else:
        residual_x=x
        stride=1
    residual = conv2d_bn_relu(x,
                              filters=filters, 
                              kernel_size=kernel_size, 
                              weight_decay=weight_decay,
                              strides=(stride,stride)
                              )
    residual = conv2d_bn(residual,
                         filters=filters,
                         kernel_size=kernel_size,
                         weight_decay=weight_decay, 
                         strides=(1,1)
                         )
    out = layers.add([residual_x, residual])
    out = Activation('relu')(out)
    return out

def ResNet18(classes,input_shape,weight_decay=1e-4):
    input=Input(shape=input_shape)
    x=input
    x=conv2d_bn_relu(x,filters=64,kernel_size=(3,3),weight_decay=weight_decay,strides=(1,1))
    
    ##conv2
    x=ResidualBlock(x,filters=64,kernel_size=(3,3),weight_decay=weight_decay,downsample=False)   
    x=ResidualBlock(x,filters=64,kernel_size=(3,3),weight_decay=weight_decay,downsample=False)   

    x=ResidualBlock(x,filters=128,kernel_size=(3,3),weight_decay=weight_decay,downsample=True)   
    x=ResidualBlock(x,filters=128,kernel_size=(3,3),weight_decay=weight_decay,downsample=False)   

    x=ResidualBlock(x,filters=256,kernel_size=(3,3),weight_decay=weight_decay,downsample=True)   
    x=ResidualBlock(x,filters=256,kernel_size=(3,3),weight_decay=weight_decay,downsample=False)   

    x=ResidualBlock(x,filters=512,kernel_size=(3,3),weight_decay=weight_decay,downsample=True)   
    x=ResidualBlock(x,filters=512,kernel_size=(3,3),weight_decay=weight_decay,downsample=False)   
    x=AveragePooling2D(pool_size=(4,4),padding='valid')(x)
    x=Flatten()(x)
    x=Dense(classes,activation='softmax')(x)
    model=Model(input,x,name='ResNet18')
    return model

def ResNetForCIFAR10(classes, input_shape, block_layers_num, weight_decay, name):
    input=Input(shape=input_shape)
    x=input
    x=conv2d_bn_relu(x,filters=16,kernel_size=(3,3),weight_decay=weight_decay,strides=(1,1))
    
    ##conv2
    for i in range(block_layers_num):
        x=ResidualBlock(x,filters=16,kernel_size=(3,3),weight_decay=weight_decay,downsample=False)   
   

    x=ResidualBlock(x,filters=32,kernel_size=(3,3),weight_decay=weight_decay,downsample=True)   
    for i in range(block_layers_num-1):
        x=ResidualBlock(x,filters=32,kernel_size=(3,3),weight_decay=weight_decay,downsample=False)   
   

    x=ResidualBlock(x,filters=64,kernel_size=(3,3),weight_decay=weight_decay,downsample=True)   
    for i in range(block_layers_num-1):
        x=ResidualBlock(x,filters=64,kernel_size=(3,3),weight_decay=weight_decay,downsample=False)   

    x=AveragePooling2D(pool_size=(8,8),padding='valid')(x)
    x=Flatten()(x)
    x=Dense(classes,activation='softmax')(x)
    model=Model(input,x,name=name)
    return model

def ResNet20ForCIFAR10(classes,input_shape,weight_decay, name='ResNet20'):
    return ResNetForCIFAR10(classes,input_shape,3,weight_decay, name)

def ResNet32ForCIFAR10(classes,input_shape,weight_decay, name='ResNet32'):
    return ResNetForCIFAR10(classes,input_shape,5,weight_decay, name)


def ResNet50ForCIFAR10(classes,input_shape,weight_decay, name='ResNe50'):
    return ResNetForCIFAR10(classes,input_shape,8,weight_decay, name)

# 모델 저장
today = date.today()
today_str = today.strftime("%Y-%m-%d-%H-%M")
today_weight_path = f"./weights/{today_str}/"

os.makedirs(today_weight_path, exist_ok=True)
file_path = today_weight_path+"epochs:{epoch:02d}-val_acc:{val_accuracy:.3f}-val_loss{val_loss:.3f}.hdf5"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    file_path, monitor='val_loss', verbose=1, save_best_only=True,
    save_weights_only=False, mode='auto'
)

# tensorboard -> $ tensorboard --logdir=./result_log/
log_dir = './result_log'
os.makedirs(log_dir, exist_ok=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1,
                                                      profile_batch=1
                                                      )
# 하이퍼 파라미터
weight_decay=1e-4
lr=1e-3

# 클래스 개수
num_classes=10

# 모델 생성 및 컴파일
resnet50=ResNet50ForCIFAR10(classes=num_classes,input_shape=(32,32,3), weight_decay=weight_decay)
opt=optimizers.Adam(lr=lr)
resnet50.compile(optimizer=opt,
                 loss=losses.categorical_crossentropy,
                 metrics=['accuracy'])

resnet50.summary()

# 조건에 맞게 감소하는 learning_rate
def lr_scheduler(epoch):
    new_lr=lr
    if epoch<=61:
        pass
    elif epoch >61 and epoch <=91:
        new_lr=lr*0.1
    elif epoch > 91:
        new_lr=lr*0.01
    print('new lr:%2e' %new_lr)
    return new_lr

reduce_lr=LearningRateScheduler(lr_scheduler)

# 데이터 셋
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Data Agumentation
datagen = ImageDataGenerator(
    rotation_range=15,  # 이미지를 최대 15도까지 회전
    width_shift_range=0.1,  # 이미지를 가로로 최대 10%까지 이동
    height_shift_range=0.1,  # 이미지를 세로로 최대 10%까지 이동
    shear_range=0.1,  # 이미지를 최대 10%까지 밀림
    zoom_range=0.1,  # 이미지를 최대 10%까지 확대/축소
    horizontal_flip=True,  # 이미지를 수평으로 뒤집기
    fill_mode="nearest",  # 새롭게 생성된 픽셀을 인접한 픽셀로 채우기
)
datagen.fit(x_train)

# 학습
resnet50.fit(
    datagen.flow(x_train, y_train, batch_size=256),
    epochs=300,
    validation_data=(x_test, y_test),
    callbacks=[reduce_lr, tensorboard_callback, model_checkpoint]
)