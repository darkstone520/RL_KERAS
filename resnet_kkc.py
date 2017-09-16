import tensorflow as tf
import numpy as np
import time

class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.class_num = 10
        self._build_net()

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0 / n)))
            return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    def _build_net(self):
        with tf.variable_scope(self.name):
            with tf.name_scope('input_layer') as scope:
                self.dropout_rate = tf.Variable(tf.constant(value=0.5), name='dropout_rate')
                self.training = tf.placeholder(tf.bool, name='training')
                self.X = tf.placeholder(tf.float32, [None, 144*144], name='x_data')
                X_img = tf.reshape(self.X, shape=[-1, 144, 144, 1])
                self.Y = tf.placeholder(tf.float32, [None, self.class_num], name='y_data')

            with tf.name_scope('conv1') as scope:

                # 144x144
                self.W1_sub = tf.get_variable(name='W1_sub', shape=[7,7,1,64], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L1_sub = tf.nn.conv2d(input=X_img, filter=self.W1_sub, strides=[1,1,1,1], padding='SAME')

            with tf.name_scope('conv2_x'):

                # output : 72x72
                self.L2_sub = tf.nn.max_pool(value=self.L1_sub, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
                self.W2_sub = tf.get_variable(name='W2_sub', shape=[3,3,64,64], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())

                # output : 72x72
                self.L2_sub = tf.nn.conv2d(input=self.L2_sub, filter=self.W2_sub, strides=[1,1,1,1], padding='SAME')
                self.L2_sub = self.BN(input=self.L2_sub, scale=True, training=self.training, name='Conv2_sub_BN_1')
                self.L2_sub = self.parametric_relu(self.L2_sub, 'R_conv2_1')

                # output : 72x72
                self.L2_sub = tf.nn.conv2d(input=self.L2_sub, filter=self.W2_sub, strides=[1,1,1,1], padding='SAME')
                self.L2_sub = self.BN(input=self.L2_sub, scale=True, training=self.training, name='Conv2_sub_BN_2')
                self.L2_sub = self.parametric_relu(self.L2_sub, 'R_conv2_2')

                # output : 72x72
                self.L2_sub = tf.nn.conv2d(input=self.L2_sub, filter=self.W2_sub, strides=[1,1,1,1], padding='SAME')
                self.L2_sub = self.BN(input=self.L2_sub, scale=True, training=self.training, name='Conv2_sub_BN_3')
                self.L2_sub = self.parametric_relu(self.L2_sub, 'R_conv2_3')

                # output : 72x72
                self.L2_sub = tf.nn.conv2d(input=self.L2_sub, filter=self.W2_sub, strides=[1,1,1,1], padding='SAME')
                self.L2_sub = self.BN(input=self.L2_sub, scale=True, training=self.training, name='Conv2_sub_BN_4')
                self.L2_sub = self.parametric_relu(self.L2_sub, 'R_conv2_4') + tf.layers.conv2d(self.L1_sub, kernel_size=(1,1), strides=(2,2), padding='SAME', filters=64, activation=tf.nn.relu)


            with tf.name_scope('conv3_x'):

                self.W3_sub = tf.get_variable(name='W3_sub', shape=[3,3,64,128], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.W3_sub_1 = tf.get_variable(name='W3_sub_1', shape=[3,3,128,128], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())

                # output = 36x36
                self.L3_sub = tf.nn.conv2d(input=self.L2_sub, filter=self.W3_sub, strides=[1,2,2,1], padding='SAME')
                self.L3_sub = self.BN(input=self.L3_sub, scale=True, training=self.training, name='Conv3_sub_BN_1')
                self.L3_sub = self.parametric_relu(self.L3_sub, 'R_conv3_1')

                # output : 36x36
                self.L3_sub = tf.nn.conv2d(input=self.L3_sub, filter=self.W3_sub_1, strides=[1, 1, 1, 1], padding='SAME')
                self.L3_sub = self.BN(input=self.L3_sub, scale=True, training=self.training, name='Conv3_sub_BN_2')
                self.L3_sub = self.parametric_relu(self.L3_sub, 'R_conv3_2')

                # output : 36x36
                self.L3_sub = tf.nn.conv2d(input=self.L3_sub, filter=self.W3_sub_1, strides=[1, 1, 1, 1], padding='SAME')
                self.L3_sub = self.BN(input=self.L3_sub, scale=True, training=self.training, name='Conv3_sub_BN_3')
                self.L3_sub = self.parametric_relu(self.L3_sub, 'R_conv3_3')

                # output : 36x36
                self.L3_sub = tf.nn.conv2d(input=self.L3_sub, filter=self.W3_sub_1, strides=[1,1,1,1], padding='SAME')
                self.L3_sub = self.BN(input=self.L3_sub, scale=True, training=self.training, name='Conv3_sub_BN_4')
                self.L3_sub = self.parametric_relu(self.L3_sub, 'R_conv3_4') + tf.layers.conv2d(self.L2_sub, kernel_size=(1,1), strides=(2,2), padding='SAME', filters=128, activation=tf.nn.relu)

            # INPUT으로 128개의 필터가 들어온다.
            with tf.name_scope('conv4_x'):

                self.W4_sub = tf.get_variable(name='W4_sub', shape=[3,3,128,256], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.W4_sub_1 = tf.get_variable(name='W4_sub_1', shape=[3,3,256,256], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())

                # 128개의 필터를 64개로 축소
                self.bottle_neck = tf.get_variable(name='W4_bottle_nack', shape=[1,1,128,64], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                # 64개로 축소된 필터를 다음 레이어로 256개로 확대
                self.bottle_neck_1 = tf.get_variable(name='W4_bottle_nack_1', shape=[1,1,64,256], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                # 64개로 축소된 필터를 3x3 연산
                self.bottle_neck_2 = tf.get_variable(name='W4_bottle_nack_2', shape=[3,3,64,64], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                # 64개 필터를 128개로 확대
                self.bottle_neck_3 = tf.get_variable(name='W4_bottle_nack_3', shape=[3,3,64,128], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())


                # output 18x18
                # 128개 필터를 64개로 축소
                self.L4_sub = tf.nn.conv2d(input=self.L3_sub, filter=self.bottle_neck, strides=[1,2,2,1], padding='SAME')
                self.L4_sub = self.BN(input=self.L4_sub, scale=True, training=self.training, name='Conv4_sub_BN_1')
                self.L4_sub = self.parametric_relu(self.L4_sub, 'R_conv4_1')

                # 64개로 축소된 필터에 대해 3x3 컨벌루션 연산
                self.L4_sub = tf.nn.conv2d(input=self.L4_sub, filter=self.bottle_neck_2, strides=[1,1,1,1], padding='SAME')
                self.L4_sub = self.BN(input=self.L4_sub, scale=True, training=self.training, name='Conv4_sub_BN_1_1')
                self.L4_sub = self.parametric_relu(self.L4_sub, 'R_conv4_1_1')

                # 3x3 연산 후 다시 필터를 128개로 바꾼다.
                self.L4_sub = tf.nn.conv2d(input=self.L4_sub, filter=self.bottle_neck_3, strides=[1,1,1,1], padding='SAME')
                self.L4_sub = self.BN(input=self.L4_sub, scale=True, training=self.training, name='Conv4_sub_BN_1_2')
                self.L4_sub = self.parametric_relu(self.L4_sub, 'R_conv4_1_2')


                # output : 18x18
                # 128개 필터를 64개로 축소
                self.L4_sub = tf.nn.conv2d(input=self.L4_sub, filter=self.bottle_neck, strides=[1, 1, 1, 1], padding='SAME')
                self.L4_sub = self.BN(input=self.L4_sub, scale=True, training=self.training, name='Conv4_sub_BN_2')
                self.L4_sub = self.parametric_relu(self.L4_sub, 'R_conv4_2')

                # output : 18x18
                # 64개로 축소된 필터에 대해 3x3 컨벌루션 연산
                self.L4_sub = tf.nn.conv2d(input=self.L4_sub, filter=self.bottle_neck_2, strides=[1, 1, 1, 1], padding='SAME')
                self.L4_sub = self.BN(input=self.L4_sub, scale=True, training=self.training, name='Conv4_sub_BN_2_1')
                self.L4_sub = self.parametric_relu(self.L4_sub, 'R_conv4_2_1')

                # output : 18x18, 256개 필터
                # 3x3 연산후 256개 필터로 바꾼다
                self.L4_sub = tf.nn.conv2d(input=self.L4_sub, filter=self.bottle_neck_1, strides=[1,1,1,1], padding='SAME')
                self.L4_sub = self.BN(input=self.L4_sub, scale=True, training=self.training, name='Conv4_sub_BN_2_2')
                self.L4_sub = self.parametric_relu(self.L4_sub, 'R_conv4_2_2') + tf.layers.conv2d(self.L3_sub, kernel_size=(1,1), strides=(2,2), padding='SAME', filters=256, activation=tf.nn.relu)


            with tf.name_scope('conv5_x'):

                self.W5_sub = tf.get_variable(name='W5_sub', shape=[3,3,256,512], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.W5_sub_1 = tf.get_variable(name='W5_sub_1', shape=[3,3,512,512], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())

                # output 9x9
                self.L5_sub = tf.nn.conv2d(input=self.L4_sub, filter=self.W5_sub, strides=[1,2,2,1], padding='SAME')
                self.L5_sub = self.BN(input=self.L5_sub, scale=True, training=self.training, name='Conv5_sub_BN_1')
                self.L5_sub = self.parametric_relu(self.L5_sub, 'R_conv5_1')

                # output : 9x9
                self.L5_sub = tf.nn.conv2d(input=self.L5_sub, filter=self.W5_sub_1, strides=[1, 1, 1, 1], padding='SAME')
                self.L5_sub = self.BN(input=self.L5_sub, scale=True, training=self.training, name='Conv5_sub_BN_2')
                self.L5_sub = self.parametric_relu(self.L5_sub, 'R_conv5_2')

                # output : 9x9
                self.L5_sub = tf.nn.conv2d(input=self.L5_sub, filter=self.W5_sub_1, strides=[1, 1, 1, 1], padding='SAME')
                self.L5_sub = self.BN(input=self.L5_sub, scale=True, training=self.training, name='Conv5_sub_BN_3')
                self.L5_sub = self.parametric_relu(self.L5_sub, 'R_conv5_3')

                # output : 9x9
                self.L5_sub = tf.nn.conv2d(input=self.L5_sub, filter=self.W5_sub_1, strides=[1,1,1,1], padding='SAME')
                self.L5_sub = self.BN(input=self.L5_sub, scale=True, training=self.training, name='Conv5_sub_BN_4')
                self.L5_sub = self.parametric_relu(self.L5_sub, 'R_conv5_4') + tf.layers.conv2d(self.L4_sub, kernel_size=(1,1), strides=(2,2), padding='SAME', filters=512, activation=tf.nn.relu)

            with tf.name_scope('avg_pool') as scope:
                self.avg_pool = tf.nn.avg_pool(value=self.L5_sub, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
                self.avg_pool = tf.reshape(self.avg_pool, shape=[-1, 9*9*512])

            with tf.name_scope('fc_layer1') as scope:
                self.W_fc1 = tf.get_variable(name='W_fc1', shape=[9*9*512, 1000], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b_fc1 = tf.Variable(tf.constant(value=0.001, shape=[1000], name='b_fc1'))
                self.L6 = tf.matmul(self.avg_pool, self.W_fc1) + self.b_fc1
                self.L6 = self.BN(input=self.L6, scale=True, training=self.training, name='Conv6_sub_BN')
                self.L_fc1 = self.parametric_relu(self.L6, 'R_fc1')

            self.W_out = tf.get_variable(name='W_out', shape=[1000, self.class_num], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b_out = tf.Variable(tf.constant(value=0.001, shape=[self.class_num], name='b_out'))
            self.logits = tf.matmul(self.L_fc1, self.W_out) + self.b_out

        ################################################################################################################
        ## ▣ L2-Regularization
        ##  ⊙ λ/(2*N)*Σ(W)²-> (0.001/(2*tf.to_float(tf.shape(self.X)[0])))*tf.reduce_sum(tf.square(self.W7))
        ################################################################################################################
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)) + (0.01/(2*tf.to_float(tf.shape(self.Y)[0])))*tf.reduce_sum(tf.square(self.W_out))

        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0025,
        #                                            momentum=0.9, decay=0.99,
        #                                            epsilon=0.01).minimize(self.cost)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0025).minimize(self.cost)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, 1), tf.arg_max(self.Y, 1)), dtype=tf.float32))

        # self.tensorflow_summary()

    ####################################################################################################################
    ## ▣ Tensorboard logging
    ##  ⊙ tf.summary.histogram : 여러개의 행렬 값을 logging 하는 경우
    ##  ⊙ tf.summary.scalar : 한개의 상수 값을 logging 하는 경우
    ####################################################################################################################
    def tensorflow_summary(self):
        self.W1_hist = tf.summary.histogram('W1_conv1', self.W1)
        self.b1_hist = tf.summary.histogram('b1_conv1', self.b1)
        self.L1_hist = tf.summary.histogram('L1_conv1', self.L1)

        self.W2_hist = tf.summary.histogram('W2_conv2', self.W2)
        self.b2_hist = tf.summary.histogram('b2_conv2', self.b2)
        self.L2_hist = tf.summary.histogram('L2_conv2', self.L2)

        self.W3_hist = tf.summary.histogram('W3_conv3', self.W3)
        self.b3_hist = tf.summary.histogram('b3_conv3', self.b3)
        self.L3_hist = tf.summary.histogram('L3_conv3', self.L3)

        self.W4_hist = tf.summary.histogram('W4_conv4', self.W4)
        self.b4_hist = tf.summary.histogram('b4_conv4', self.b4)
        self.L4_hist = tf.summary.histogram('L4_conv4', self.L4)

        self.W5_hist = tf.summary.histogram('W5_conv5', self.W5)
        self.b5_hist = tf.summary.histogram('b5_conv5', self.b5)
        self.L5_hist = tf.summary.histogram('L5_conv5', self.L5)

        self.W_fc1_hist = tf.summary.histogram('W6_fc1', self.W_fc1)
        self.b_fc1_hist = tf.summary.histogram('b6_fc1', self.b_fc1)
        self.L_fc1_hist = tf.summary.histogram('L6_fc1', self.L_fc1)

        self.W_fc2_hist = tf.summary.histogram('W6_fc2', self.W_fc2)
        self.b_fc2_hist = tf.summary.histogram('b6_fc2', self.b_fc2)
        self.L_fc2_hist = tf.summary.histogram('L6_fc2', self.L_fc2)

        self.cost_hist = tf.summary.scalar(self.name+'/cost_hist', self.cost)
        self.accuracy_hist = tf.summary.scalar(self.name+'/accuracy_hist', self.accuracy)

        # ※ merge_all 로 하는 경우, hist 를 모으지 않는 변수들도 대상이 되어서 에러가 발생한다.
        #    따라서 merge 로 모으고자하는 변수를 각각 지정해줘야한다.
        self.merged = tf.summary.merge([self.W1_hist, self.b1_hist, self.L1_hist,
                                        self.W2_hist, self.b2_hist, self.L2_hist,
                                        self.W3_hist, self.b3_hist, self.L3_hist,
                                        self.W4_hist, self.b4_hist, self.L4_hist,
                                        self.W5_hist, self.b5_hist, self.L5_hist,
                                        self.W_fc1_hist, self.b_fc1_hist, self.L_fc1_hist,
                                        self.W_fc2_hist, self.b_fc2_hist, self.L_fc2_hist,
                                        self.cost_hist, self.accuracy_hist])

    def predict(self, x_test):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: False})

    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: False})

    def train(self, x_data, y_data):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.training: True})

    ####################################################################################################################
    ## ▣ Parametric Relu or Leaky Relu
    ##  ⊙ alpha 값을 설정해 0 이하인 경우 alpha 만큼의 경사를 설정해서 0 이 아닌 값을 리턴하는 함수
    ##  ⊙ Parametric Relu : 학습을 통해 최적화된 alpha 값을 구해서 적용하는 Relu 함수
    ##  ⊙ Leaky Relu      : 0 에 근접한 작은 값을 alpha 값으로 설정하는 Relu 함수
    ####################################################################################################################
    def parametric_relu(self, _x, name):
        alphas = tf.get_variable(name,
                                 _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.01),
                                 dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5
        return pos + neg

    ####################################################################################################################
    ## ▣ Maxout
    ##  ⊙ Convolution 계층이나 FC 계층에서 활성화 함수 대신 dropout 의 효율을 극대화하기 위해 사용하는 함수
    ##  ⊙ conv 또는 affine 계층을 거친 값들에 대해 k 개씩 그룹핑을 수행하고 해당 그룹내에서 가장 큰 값을 다음 계층으로
    ##     보내는 기법
    ####################################################################################################################
    def max_out(self, inputs, num_units, axis=None):
        shape = inputs.get_shape().as_list()
        if shape[0] is None:
            shape[0] = -1
        if axis is None:  # Assume that channel is the last dimension
            axis = -1
        num_channels = shape[axis]
        if num_channels % num_units:
            raise ValueError(
                'number of features({}) is not a multiple of num_units({})'.format(num_channels, num_units))
        shape[axis] = num_units  # m
        shape += [num_channels // num_units]  # k
        outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
        return outputs

    ####################################################################################################################
    ## ▣ Batch Normalization
    ##  ⊙ training 하는 과정 자체를 전체적으로 안정화하여 학습 속도를 가속시킬 수 있는 방법
    ##  ⊙ Network의 각 층이나 Activation 마다 input_data 의 distribution 을 평균 0, 표준편차 1인 input_data로 정규화시키는 방법
    ##  ⊙ 초기 파라미터 --> beta : 0 , gamma : 1 , decay : 0.99 , epsilon : 0.001
    ####################################################################################################################
    def BN(self, input, training, scale, name, decay=0.99):
        return tf.contrib.layers.batch_norm(input, decay=decay, scale=scale, is_training=training, updates_collections=None, scope=name)

    ####################################################################################################################
    ## ▣ dynamic_learning
    ##  ⊙ epoch 가 클수록 아니면 early_stopping 이 시작되면 점차적으로 learning_rate의 값을 줄여 안정적인 훈련이 가능한 방법
    ####################################################################################################################
    def dynamic_learning(self,learning_rate,earlystop,epoch):
        max_learning_rate = learning_rate
        min_learing_rate = 0.001
        learning_decay = 60 # 낮을수록 빨리 떨어진다.
        if earlystop >= 1:
            lr = min_learing_rate + (max_learning_rate - min_learing_rate) * np.exp(-epoch / learning_decay)
        else:
            lr = max_learning_rate
        return round(lr,4)