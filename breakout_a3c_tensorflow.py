
import tensorflow as tf
import numpy as np
import threading
import gym
import os
from skimage.transform import resize
from skimage.color import rgb2gray


def copy_src_to_dst(from_scope, to_scope):
    """Creates a copy variable weights operation

    Args:
        from_scope (str): The name of scope to copy from
            It should be "global"
        to_scope (str): The name of scope to copy to
            It should be "thread-{}"

    Returns:
        list: Each element is a copy operation
    """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder
# 학습속도를 높이기 위해 흑백화면으로 전처리
# 흑백으로 바꾼 후 255를 곱해서 연산이 빠르도록 int값으로 바꿈
def pre_processing(observe):
    observe = crop_image(observe, height_range=(35,195))
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe

def crop_image(image, height_range=(35, 195)):
    """Crops top and bottom

    Args:
        image (3-D array): Numpy image (H, W, C)
        height_range (tuple): Height range between (min_height, max_height)
            will be kept

    Returns:
        image (3-D array): Numpy image (max_H - min_H, W, C)
    """
    h_beg, h_end = height_range
    return image[h_beg:h_end, ...]

def discount_reward(rewards, deads, gamma=0.99):
    """Returns discounted rewards

    Args:
        rewards (1-D array): Reward array
        gamma (float): Discounted rate

    Returns:
        discounted_rewards: same shape as `rewards`

    Notes:
        In Pong, when the reward can be {-1, 0, 1}.
        However, when the reward is either -1 or 1,
        it means the game has been reset.
        Therefore, it's necessaray to reset `running_add` to 0
        whenever the reward is nonzero
    """
    discounted_r = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        if deads[t] == True:
            running_add = 0
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add

    return discounted_r


class A3CNetwork(object):

    def __init__(self, name, input_shape, output_dim, logdir=None):
        """A3C Network tensors and operations are defined here

        Args:
            name (str): The name of scope
            input_shape (list): The shape of input image [H, W, C]
            output_dim (int): Number of actions
            logdir (str, optional): directory to save summaries

        Notes:
            You should be familiar with Policy Gradients.
            The only difference between vanilla PG and A3C is that there is
            an operation to apply gradients manually
        """
        with tf.variable_scope(name):
            xavier = tf.contrib.layers.xavier_initializer()
            conv2d_initializer = tf.contrib.layers.xavier_initializer_conv2d()

            self.states = tf.placeholder(tf.float32, shape=[None, *input_shape], name="states")
            self.actions = tf.placeholder(tf.uint8, shape=[None], name="actions")
            self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")
            self.advantage = tf.placeholder(tf.float32, shape=[None], name="advantage")

            action_onehot = tf.one_hot(self.actions, output_dim, name="action_onehot")
            net = self.states

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
                                      units=256,
                                      kernel_initializer=xavier,
                                      name='dense',
                                      activation=tf.nn.relu)


            # actor network
            actions = tf.layers.dense(net, output_dim, name="final_fc")
            self.action_prob = tf.nn.softmax(actions, name="action_prob")
            single_action_prob = tf.reduce_sum(self.action_prob * action_onehot, axis=1)

            entropy = - self.action_prob * tf.log(self.action_prob + 1e-7)
            entropy = tf.reduce_sum(entropy, axis=1)

            log_action_prob = tf.log(single_action_prob + 1e-7)
            maximize_objective = log_action_prob * self.advantage + entropy * 0.005
            self.actor_loss = - tf.reduce_mean(maximize_objective)

            # value network
            self.values = tf.squeeze(tf.layers.dense(net, 1, name="values"))
            self.value_loss = tf.losses.mean_squared_error(labels=self.rewards,
                                                           predictions=self.values)

            self.total_loss = self.actor_loss + self.value_loss * .5
            self.optimizer = tf.train.RMSPropOptimizer(decay=0, momentum=0.9, epsilon=0.01, learning_rate=0.00025)

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        self.gradients = self.optimizer.compute_gradients(self.total_loss, var_list)
        self.gradients_placeholders = []

        for grad, var in self.gradients:
            placeholder = tf.placeholder(var.dtype, shape=var.get_shape())
            placeholder = tf.clip_by_norm(placeholder, 40)
            self.gradients_placeholders.append((placeholder, var))
        self.apply_gradients = self.optimizer.apply_gradients(self.gradients_placeholders)

        if logdir:
            loss_summary = tf.summary.scalar("total_loss", self.total_loss)
            value_summary = tf.summary.histogram("values", self.values)

            self.summary_op = tf.summary.merge([loss_summary, value_summary])
            self.summary_writer = tf.summary.FileWriter(logdir)


class Agent(threading.Thread):

    def __init__(self, session, env, coord, name, global_network, input_shape, output_dim, logdir=None):
        """Agent worker thread

        Args:
            session (tf.Session): Tensorflow session needs to be shared
            env (gym.Env): Gym environment (Pong-v0)
            coord (tf.train.Coordinator): Tensorflow Queue Coordinator
            name (str): Name of this worker
            global_network (A3CNetwork): Global network that needs to be updated
            input_shape (list): Required for local A3CNetwork, [H, W, C]
            output_dim (int): Number of actions
            logdir (str, optional): If logdir is given, will write summary

        Methods:
            print(reward): prints episode rewards
            play_episode(): a single episode logic is stored in here
            run(): override threading.Thread.run
            choose_action(state)
            train(states, actions, rewards)
        """
        super(Agent, self).__init__()
        self.local = A3CNetwork(name, input_shape, output_dim, logdir)
        self.global_to_local = copy_src_to_dst("global", name)
        self.global_network = global_network

        self.input_shape = input_shape
        self.output_dim = output_dim
        self.env = env
        self.sess = session
        self.coord = coord
        self.name = name
        self.logdir = logdir

    def print(self, reward):
        message = "Agent(name={}, reward={})".format(self.name, reward)
        print(message)

    def play_episode(self):
        self.sess.run(self.global_to_local)

        states = []
        actions = []
        rewards = []
        deads = []
        s = self.env.reset()
        s = pre_processing(s)
        history = np.stack((s, s, s, s), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        done = False
        dead = False
        total_reward = 0
        time_step = 0
        _, _, _, info = self.env.step(1)
        start_lives = info['ale.lives']

        while not done:

            a = self.choose_action(history)

            if a == 0:
                real_action = 1
            elif a == 1:
                real_action = 2
            elif a == 2:
                real_action = 3

            s2, r, done, info = self.env.step(real_action)

            if start_lives > info['ale.lives']:
                start_lives -= 1
                dead = True

            s2 = pre_processing(s2)
            next_state = np.reshape([s2], (1, 84, 84, 1))
            history = np.append(next_state, history[:, :, :, :3], axis=3)
            total_reward += r

            if done:
                dead = True

            states.append(history)
            actions.append(a)
            rewards.append(r)
            deads.append(dead)


            if dead:
                dead = False


            if r == -1 or r == 1 or done:
                time_step += 1

                if time_step >= 5 or done:
                    self.train(states, actions, rewards, deads)
                    self.sess.run(self.global_to_local)
                    states, actions, rewards, deads = [], [], [], []
                    time_step = 0

        self.print(total_reward)

    def run(self):
        while not self.coord.should_stop():
            self.play_episode()

    def choose_action(self, state):

        state = np.reshape(state, [-1, 84, 84, 4])
        feed = {
            self.local.states: state
        }

        action = self.sess.run(self.local.action_prob, feed)
        action = np.squeeze(action)

        return np.random.choice(np.arange(self.output_dim), p=action)

    def train(self, states, actions, rewards, deads):
        states = np.array(states)
        states = np.reshape([states], (-1,84,84,4))
        print(states.shape)
        actions = np.array(actions)
        rewards = np.array(rewards)
        deads = np.array(deads)

        feed = {
            self.local.states: states
        }

        values = self.sess.run(self.local.values, feed)

        rewards = discount_reward(rewards, deads, gamma=0.99)
        rewards = np.clip(rewards, 0., 1.)
        advantage = rewards - values
        advantage -= np.mean(advantage)
        advantage /= np.std(advantage) + 1e-8

        feed = {
            self.local.states: states,
            self.local.actions: actions,
            self.local.rewards: rewards,
            self.local.advantage: advantage
        }

        gradients = self.sess.run(self.local.gradients, feed)

        feed = []
        for (grad, _), (placeholder, _) in zip(gradients, self.global_network.gradients_placeholders):
            feed.append((placeholder, grad))

        feed = dict(feed)
        self.sess.run(self.global_network.apply_gradients, feed)


def main():

    coord = tf.train.Coordinator()
    monitor_dir = "monitors"

    n_threads = 8
    input_shape = [84, 84, 4]
    output_dim = 3  # {1, 2, 3}
    global_network = A3CNetwork(name="global",
                                input_shape=input_shape,
                                output_dim=output_dim)

    thread_list = []
    env_list = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for id in range(n_threads):
            env = gym.make("BreakoutDeterministic-v4")

            if id == 0:
                env = gym.wrappers.Monitor(env, monitor_dir, force=True)

            single_agent = Agent(env=env,
                                 session=sess,
                                 coord=coord,
                                 name="thread_{}".format(id),
                                 global_network=global_network,
                                 input_shape=input_shape,
                                 output_dim=output_dim)
            thread_list.append(single_agent)
            env_list.append(env)


        for t in thread_list:
            t.start()

        print("Ctrl + C to close")
        coord.wait_for_stop()




if __name__ == '__main__':
    main()
