# -*-encoding:utf-8 -*-
from keras.layers import concatenate
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.layers.core import Activation
from keras.layers import Multiply
from keras import regularizers


def circRNA_dnn():
    inputs = Input(shape=(2597,))
    x = Dense(1024)(inputs)
    x = Activation("relu")(x)
    x = Dense(256)(x)
    x = Activation("relu")(x)
    x = Dense(32)(x)
    x = Activation("relu")(x)
    model = Model(inputs, x)

    return model


def disease_dnn():
    inputs = Input(shape=(67,))
    x = Dense(128)(inputs)
    x = Activation("relu")(x)
    x = Dense(64)(x)
    x = Activation("relu")(x)
    x = Dense(32)(x)
    x = Activation("relu")(x)
    model = Model(inputs, x)
    return model


def get_model():
    dnn1 = circRNA_dnn()
    dnn2 = disease_dnn()
    x3 = Multiply()([dnn1.output, dnn2.output])
    combinedInput = concatenate([dnn1.output,  x3, dnn2.output])
    x = Dense(128, activation="relu", kernel_regularizer=regularizers.l1(0.001))(combinedInput)
    x = Dense(64, activation="relu", kernel_regularizer=regularizers.l1(0.001))(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=[dnn1.input, dnn2.input], outputs=x)
    return model

