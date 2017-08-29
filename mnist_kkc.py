# Lab 11 MNIST and Deep learning CNN
# https://www.tensorflow.org/tutorials/layers
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

# hyper parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 100


class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

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
            self.X = tf.placeholder(tf.float32, [None, 784])
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            with tf.variable_scope("Conv1"):
                net = tf.layers.conv2d(inputs=X_img,
                                         filters=32,
                                         kernel_size=[3, 3],
                                         padding="SAME",
                                         kernel_initializer=conv_init
                                         )

                net = slim.batch_norm(net,
                                      is_training=self.training,
                                      activation_fn=tf.nn.relu
                                      )

                net = tf.layers.max_pooling2d(inputs=net,
                                                pool_size=[2, 2],
                                                padding="SAME",
                                                strides=2
                                                )


            # Convolutional Layer #2 and Pooling Layer #2
            with tf.variable_scope("Conv2"):
                net = tf.layers.conv2d(inputs=net,
                                         filters=64,
                                         kernel_size=[3, 3],
                                         padding="SAME",
                                         kernel_initializer=conv_init
                                         )

                net = slim.batch_norm(net,
                                      is_training=self.training,
                                      activation_fn=tf.nn.relu
                                      )

                net = tf.layers.max_pooling2d(inputs=net,
                                                pool_size=[2, 2],
                                                padding="SAME",
                                                strides=2
                                                )

                net = slim.batch_norm(net,
                                      is_training=self.training,
                                      )

            # Convolutional Layer #3 and Pooling Layer #3
            with tf.variable_scope("Conv3"):
                net = tf.layers.conv2d(inputs=net,
                                         filters=128,
                                         kernel_size=[3, 3],
                                         padding="SAME",
                                         kernel_initializer=conv_init,
                                         )

                net = slim.batch_norm(net,
                                      is_training=self.training,
                                      activation_fn=tf.nn.relu
                                      )

                net = tf.layers.max_pooling2d(inputs=net,
                                                pool_size=[2, 2],
                                                padding="SAME",
                                                strides=2
                                                )



            # Dense Layer with Relu
            with tf.variable_scope("Dense1"):
                net = tf.reshape(net, [-1, 128 * 4 * 4])
                net = tf.layers.dense(inputs=net,
                                         units=625,
                                         kernel_initializer=he_normal,
                                         activation=tf.nn.relu)


            # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
            self.logits = tf.layers.dense(inputs=net, units=10)

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.cost)

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

# initialize
sess = tf.Session()

models = []
num_models = 2
for m in range(num_models):
    models.append(Model(sess, "model" + str(m)))

sess.run(tf.global_variables_initializer())

print('Learning Started!')

# train my model
for epoch in range(training_epochs):
    avg_cost_list = np.zeros(len(models))
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        # train each model
        for m_idx, m in enumerate(models):
            c, _ = m.train(batch_xs, batch_ys)
            avg_cost_list[m_idx] += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost_list)

print('Learning Finished!')

# Test model and check accuracy
test_size = len(mnist.test.labels)
predictions = np.zeros(test_size * 10).reshape(test_size, 10)
for m_idx, m in enumerate(models):
    print(m_idx, 'Accuracy:', m.get_accuracy(
        mnist.test.images, mnist.test.labels))
    p = m.predict(mnist.test.images)
    predictions += p

ensemble_correct_prediction = tf.equal(
    tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1))
ensemble_accuracy = tf.reduce_mean(
    tf.cast(ensemble_correct_prediction, tf.float32))
print('Ensemble accuracy:', sess.run(ensemble_accuracy))
