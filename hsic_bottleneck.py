#coding=utf8
"""

# Author : rui
# Created Time : Aug 28 22:02:56 2019
# Description:
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals


from keras import optimizers
import keras.backend as K
import tensorflow as tf


def kernel_matrix(x, sigma):
    ndim = K.ndim(x)
    x1 = K.expand_dims(x, 0)
    x2 = K.expand_dims(x, 1)
    axis = tuple(range(2, ndim+1))
    return K.exp(-0.5*K.sum(K.pow(x1-x2, 2), axis=axis) / sigma ** 2)


def hsic(Kx, Ky, m):
    Kxy = K.dot(Kx, Ky)
    h = tf.linalg.trace(Kxy) / m ** 2 + K.mean(Kx) * K.mean(Ky) - \
        2 * K.mean(Kxy) / m
    return h * (m / (m-1))**2

class HSICBottleneckTrained(object):
    def __init__(self, model, batch_size, lambda_0, sigma):
        self.batch_size = batch_size
        input_x = model._feed_inputs[0]
        input_y = model._feed_targets[0]

        Kx = kernel_matrix(input_x, sigma)
        Ky = kernel_matrix(input_y, sigma)


        param2grad = {
        }
        trainable_params = []
        total_loss = 0.
        for layer in model.layers:
            if layer.name.startswith("hsic"):
                params = layer.trainable_weights
                if not params:
                    continue
                hidden_z = layer.output

                Kz = kernel_matrix(hidden_z, sigma)
                loss = hsic(Kz, Kx, batch_size) - lambda_0 * hsic(Kz, Ky, batch_size)
                total_loss += loss
                trainable_params.extend(params)
                grads = K.gradients(loss, params)
                for p, g in zip(params, grads):
                    param2grad[p.name] = g
            else:
                layer.trainable = False
        model._collected_trainable_weights = trainable_params
        model.total_loss = total_loss
        optim = model.optimizer
        def get_gradients(loss, params):
            grads = [param2grad[p.name] for p in params]
            if hasattr(self, 'clipnorm') and self.clipnorm > 0:
                norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
                grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
            if hasattr(self, 'clipvalue') and self.clipvalue > 0:
                grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
            return grads
        optim.get_gradients = get_gradients

        self.model = model

    def reshape(self, x):
        shape = list(K.int_shape(x))
        shape[0] = self.batch_size
        return K.reshape(x, tuple(shape))

    def __call__(self):
        return self.model

class PostTrained(object):
    def __init__(self, model):
        for layer in model.layers:
            if layer.name != "output_layer":
                layer.trainable = True
            else:
                # 冻结所有非输出层为不可训练
                layer.trainable = False
        self.model = model

    def __call__(self):
        return model


# 测试
if __name__ == "__main__":
    import keras.layers as L
    from keras import models
    import numpy as np

    X = np.random.standard_normal((256*400, 25))
    y = np.uint8(np.sum(X ** 2, axis=-1) > 25.)
    num_train = 256 * 360
    X_train = X[:num_train, :]
    y_train = y[:num_train]
    X_test  = X[num_train:, :]
    y_test  = y[num_train:]

    input_x = L.Input(shape=(25,))
    z1      = L.Dense(40, name="hsic_dense_1", activation="relu")(input_x)
    z2      = L.Dense(64, name="hsic_dense_2", activation="relu")(z1)
    z2      = L.Dropout(0.2)(z2)
    z3      = L.Dense(32, name="hsic_dense_3", activation="relu")(z2)
    output_x = L.Dense(1, name="output_layer", activation="sigmoid")(z3)

    model = models.Model(inputs=input_x, outputs=output_x)


    model.compile(optimizers.SGD(0.001),
                  loss="binary_crossentropy",
                  metrics=["acc"])
    model = HSICBottleneckTrained(model, batch_size=256, lambda_0=100., sigma=10.)()
    model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=256)


    model = PostTrained(model)()
    model.compile(optimizers.SGD(0.1),
                  loss="binary_crossentropy",
                  metrics=["acc"])
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=256)


del division
del print_function
del absolute_import
del unicode_literals
