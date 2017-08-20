import numpy as np
import tensorflow as tf
import os


class DQN:
    def __init__(self, session: tf.Session, input_dim, output_size: int, name: str = "main",
                 checkpoint_dir="checkpoint") -> None:

        self.session = session
        self.input_dim = input_dim
        self.output_size = output_size
        self.net_name = name
        self.checkpoint_dir = checkpoint_dir
        self._build_network()

        if checkpoint_dir is not None:
            self.saver = tf.train.Saver()
            maybe_path = os.path.join(self.checkpoint_dir, "{}_model.ckpt".format(self.net_name))
            if os.path.exists(self.checkpoint_dir) and tf.train.checkpoint_exists(maybe_path):
                print("Restored {}".format(maybe_path))
                sess = tf.get_default_session()
                self.saver.restore(sess, maybe_path)
            else:
                print("No model is found")
                os.makedirs(checkpoint_dir, exist_ok=True)

        # 텐서보드 설정
        self.sess = tf.InteractiveSession()

        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            'summary/dqn', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def _build_network(self, h_size=512, l_rate=0.00025) -> None:

        # initializer 선택
        conv2d_initializer = tf.contrib.layers.xavier_initializer_conv2d()
        xavier = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope(self.net_name):
            self.X = tf.placeholder(tf.float32, [None, *self.input_dim], name="input_x")
            self.Y = tf.placeholder('float', [None])
            self.a = tf.placeholder('int64', [None])

            if self.net_name == 'main':
                self.training = True
            else:
                self.training = False
            net = self.X

            with tf.variable_scope("layer1"):
                net = tf.layers.conv2d(net,
                                       filters=32,
                                       kernel_size=(8, 8),
                                       strides=(4, 4),
                                       name="conv",
                                       kernel_initializer=conv2d_initializer,
                                       activation=tf.nn.relu
                                       )

            with tf.variable_scope("layer2"):
                net = tf.layers.conv2d(net,
                                       filters=64,
                                       kernel_size=(4, 4),
                                       strides=(2, 2),
                                       name="conv",
                                       kernel_initializer=conv2d_initializer,
                                       activation=tf.nn.relu
                                       )

            with tf.variable_scope("layer3"):
                net = tf.layers.conv2d(net,
                                       filters=64,
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       name="conv",
                                       kernel_initializer=conv2d_initializer,
                                       activation=tf.nn.relu
                                       )

            with tf.variable_scope("fc1"):
                net = tf.contrib.layers.flatten(net)
                net = tf.layers.dense(inputs=net,
                                      units=h_size,
                                      kernel_initializer=xavier,
                                      name='dense',
                                      activation=tf.nn.relu)

            with tf.variable_scope("fc2"):
                net = tf.layers.dense(inputs=net,
                                      units=self.output_size,
                                      kernel_initializer=xavier,
                                      name='dense'
                                      )
            self.Qpred = net

        a_one_hot = tf.one_hot(self.a, self.output_size, 1.0, 0.0)
        q_val = tf.reduce_sum(tf.multiply(self.Qpred, a_one_hot), reduction_indices=1)

        # error를 -1~1 사이로 클립
        error = tf.abs(self.Y - q_val)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        self.loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)
        #optimizer = tf.train.MomentumOptimizer(learning_rate=l_rate, momentum=0.9)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=l_rate, epsilon=0.01, decay=0., momentum=0.9)
        self.train = optimizer.minimize(self.loss)

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)

        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def predict(self, state: np.ndarray) -> np.ndarray:

        return self.session.run(self.Qpred, feed_dict={self.X: state})

    def replay_train(self, targetDQN, train_batch: list) -> float:

        DISCOUNT_RATE = 0.99

        states = np.vstack([x[0] / 255. for x in train_batch])
        actions = np.array([x[1] for x in train_batch])
        rewards = np.array([x[2] for x in train_batch])

        #reward가 모두 0이 아닌 경우 min,max 정규화한다.
        # min = np.min(rewards)
        # max = np.max(rewards)
        # if np.sum(rewards != 0) != 0:
        #     rewards -= min
        #     rewards /= (max - min)

        next_states = np.vstack([x[3] / 255. for x in train_batch])
        dead = np.array([x[4] for x in train_batch])
        X = states
        Q_target = rewards + DISCOUNT_RATE * np.max(targetDQN.predict(next_states), axis=-1) * ~dead

        # self.avg_loss += self.session.run(self.train, feed_dict={self.X: X, self.Y: Q_target, self.a :actions })
        loss, _ = self.session.run([self.loss, self.train], feed_dict={self.X: X, self.Y: Q_target, self.a: actions})
        self.avg_loss += loss

    def save(self, episode) -> None:
        print("model save")
        path = os.path.join(self.checkpoint_dir, "{}_{}_model.ckpt".format(self.net_name, episode))
        self.saver.save(self.session, path)



