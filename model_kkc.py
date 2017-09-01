# Lab 11 MNIST and Deep learning CNN
# https://www.tensorflow.org/tutorials/layers
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()
        self.learning_rate = 0.0001


    def BN(self, input, training, scale, name, activation_fn=tf.nn.relu, decay=0.99):
        return tf.contrib.layers.batch_norm(input, decay=decay, scale=scale, is_training=training,
                                            updates_collections=None, scope=name, activation_fn=activation_fn)

    def _build_net(self):

        with tf.variable_scope(self.name):

            # variance_scaling_initializer의 파라미터
            # To get Convolutional Architecture for Fast Feature Embedding, use:
            # factor=1.0 mode='FAN_IN' uniform=True
            # To get Delving Deep into Rectifiers, use (Default):
            # factor=2.0 mode='FAN_IN' uniform=False
            # To get Understanding the difficulty of training deep feedforward neural networks, use:
            # factor=1.0,mode='FAN_AVG',uniform=True

            he_normal = tf.contrib.layers.variance_scaling_initializer()
            conv_init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True)

            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            self.training = tf.placeholder(tf.bool)
            self.X = tf.placeholder(tf.float32, [None, 144*144])
            X_img = tf.reshape(self.X, [-1, 144, 144, 1])
            self.Y = tf.placeholder(tf.float32, [None, 2])

            with tf.variable_scope("Conv1"):
                net = tf.layers.conv2d(inputs=X_img,
                                         filters=32,
                                         kernel_size=[3, 3],
                                         strides=[1,1],
                                         padding="valid",
                                         kernel_initializer=conv_init
                                         )

                self.BN(net, self.training, scale=True, name="Conv1_BN")

                net = tf.layers.max_pooling2d(inputs=net,
                                                pool_size=[2, 2],
                                                strides=[2,2],
                                                padding="valid",
                                                )


            # Convolutional Layer #2 and Pooling Layer #2
            with tf.variable_scope("Conv2"):
                net = tf.layers.conv2d(inputs=net,
                                         filters=64,
                                         kernel_size=[3, 3],
                                         strides=[1,1],
                                         padding="valid",
                                         kernel_initializer=conv_init
                                         )

                self.BN(net, self.training, scale=True, name="Conv2_BN")


                net = tf.layers.max_pooling2d(inputs=net,
                                                pool_size=[2, 2],
                                                padding="valid",
                                                strides=[2,2]
                                                )

            # Convolutional Layer #3 and Pooling Layer #3
            with tf.variable_scope("Conv3"):
                net = tf.layers.conv2d(inputs=net,
                                         filters=128,
                                         kernel_size=[4, 4],
                                         strides=[1,1],
                                         padding="SAME",
                                         kernel_initializer=conv_init,
                                         )

                self.BN(net, self.training, scale=True, name="Conv3_BN")


                net = tf.layers.max_pooling2d(inputs=net,
                                                pool_size=[2, 2],
                                                padding="SAME",
                                                strides=[2,2]
                                                )

            # Dense Layer with Relu
            with tf.variable_scope("Dense1"):
                net = tf.reshape(net, [-1, 128 * 15 * 15])
                net = tf.layers.dense(inputs=net,
                                      units=256,
                                      kernel_initializer=he_normal,
                                      activation=tf.nn.relu
                                      )


            # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
            self.logits = tf.layers.dense(inputs=net, units=2)

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})


