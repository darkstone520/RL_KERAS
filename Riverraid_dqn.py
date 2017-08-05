from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten
from keras.optimizers import RMSprop
from keras.models import Sequential
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque
from keras import backend as K
import tensorflow as tf
import numpy as np
import random
import gym
import matplotlib.pyplot as plt

EPISODES = 10000


class DQNAgent:

    def __init__(self, action_size):
        self.render = False
        self.load_model = False
        self.action_size = action_size
        self.state_size = (84,84,4)
        self.epsilon = 1.0
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 1000000.
        self.epsilon_dacay_step = (self.epsilon_start - self.epsilon_end) / self.exploration_steps
        self.batch_size = 32
        self.train_start = 100#50000
        self.target_update_frequency = 10
        self.replay_memory = deque(maxlen=400000)
        self.discount_factor = 0.99
        self.no_op_steps = 30
        self.train_model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.avg_q_max, self.avg_loss = 0, 0

        self.user_optimizer = self.optimizer()

        # 텐서보드 설정
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            'summary/Riverraid_dqn', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        if self.load_model:
            self.train_model.load_weights("./save_model/Riverraid_dqn.h5")


    def build_model(self):

        model = Sequential()

        model.add(Conv2D(32, (8, 8), strides=(4,4), input_shape=self.state_size, activation="relu"))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))

        # optimizer를 따로 정의 하지 않을 때

        # adam = Adam(lr=1e-6)
        # model.compile(loss='mse', optimizer=adam)
        # print("We finish building the model")

        model.summary()

        return model


    def get_action(self, history):
        history = history/255.
        if np.random.randn() <= self.epsilon_dacay_step:
            action = np.random.randint(self.action_size)
        else:
            q_value = self.train_model.predict(history)
            action = np.argmax(q_value, axis=-1)

        return action


    def optimizer(self):
        action = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        # train_model이 predict 한 q_value
        prediction = self.train_model.output

        # action값은 1개의 int 값으로 return 되므로 one_hot_encoding 한다.
        action_one_hot = K.one_hot(action, self.action_size)
        q_value = K.sum(prediction * action_one_hot, axis=-1)

        # 후로버스로 err를 clip한 loss 식
        err = K.abs(y-q_value)
        quadratic_part = K.clip(err, 0.0, 1.0)
        linear_part = err - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        optimizer = RMSprop(lr=0.00025, epsilon=0.01)
        updates = optimizer.get_updates(self.train_model.trainable_weights, [], loss)
        train = K.function([self.train_model.input, action, y], [loss], updates=updates)

        return train



    def update_target_model(self):
        self.target_model.set_weights(self.train_model.get_weights())


    def append_sample(self, history, action, reward, next_history, dead):
        self.replay_memory.append((history, action, reward, next_history, dead))

    def train(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_dacay_step

        mini_batch = random.sample(self.replay_memory, self.batch_size)

        # shape = (batch_size, 84, 84, 4)
        # agent.append_sample(history, action, reward, next_history, dead)
        history = np.vstack([x[0]/255. for x in mini_batch])
        action = np.array([x[1] for x in mini_batch])
        reward = np.array([x[2] for x in mini_batch])
        next_history = np.vstack([x[3]/255. for x in mini_batch])
        dead = np.array([x[4] for x in mini_batch])

        target = reward + self.discount_factor * np.max(self.target_model.predict(next_history),axis=-1) * ~dead
        #print(reward)
        #print(self.target_model.predict(next_history))
        #print(target)

        # loss = (정답-예측)^2
        loss = self.user_optimizer([history, action, target])
        #print(loss)
        self.avg_loss += loss[0]


        # 각 에피소드 당 학습 정보를 기록

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

def crop_image(image, height_range=(2, 162), width_range=(8,160)):
    """Crops top and bottom

    Args:
        image (3-D array): Numpy image (H, W, C)
        height_range (tuple): Height range between (min_height, max_height)
            will be kept

    Returns:
        image (3-D array): Numpy image (max_H - min_H, W, C)
    """
    h_beg, h_end = height_range
    w_beg, w_end = width_range
    return image[h_beg:h_end, w_beg:w_end]


def pre_processing(observe):
    observe = crop_image(observe)
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe

def plot_image(image):
    image = np.squeeze(image)
    shape = image.shape

    if len(shape) == 2:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)

    plt.show()


if __name__ == "__main__":
    # 환경과 DQN 에이전트 생성
    env = gym.make('RiverraidDeterministic-v4')
    env = gym.wrappers.Monitor(env, directory="gym-results/", force=True)
    agent = DQNAgent(action_size=3)

    scores, episodes, global_step = [], [], 0

    for e in range(EPISODES):
        done = False
        dead = False

        step, score, start_life = 0, 0, 4
        observe = env.reset()

        for _ in range(random.randint(1, agent.no_op_steps)):
            observe, _, _, _ = env.step(5) # 발사 없이 느린 전진


        state = pre_processing(observe)
        history = np.stack((state, state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 5))

        while not done:
            if agent.render:
                env.render()
            global_step += 1
            step += 1

            action = agent.get_action(history[:,:,:,:4])
            # 목숨은 4개
            # 0 정지, 1 발사 및 전진, 2 발사없이 빠른 전진, 3 오른쪽, 4 왼쪽 , 5 발사없이 느린 전진
            # 목표물 30, 60, 80
            if action == 0:
                real_action = 1
            elif action == 1:
                real_action = 4
            else:
                real_action = 3

            observe, reward, done, info = env.step(real_action)

            if reward == 30:
                reward = 1
            elif reward == 60:
                reward = 2
            elif reward == 80:
                reward = 3
            elif reward < 0:
                reward =0
            elif reward >= 81:
                reward = 3


            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            #plot_image(observe)
            next_state = pre_processing(observe)
            history[:, :, :, 4] = next_state
            #plot_image(history[:,:,:,4])

            agent.avg_q_max += np.amax(
                agent.train_model.predict(np.float32(history[:,:,:,:4] / 255.))[0])

            agent.append_sample(history[:,:,:,:4], action, reward, history[:,:,:,1:], dead)

            if len(agent.replay_memory) >= agent.train_start:
                agent.train()

                if global_step % agent.target_update_frequency ==0:
                    agent.update_target_model()

            score += reward

            if dead:
                dead = False
            else:
                history[:,:,:,:4] = history[:,:,:,1:]

            if done:
                # 각 에피소드 당 학습 정보를 기록
                if global_step > agent.train_start:
                    stats = [score, agent.avg_q_max / float(step), step,
                             agent.avg_loss / float(step)]
                    for i in range(len(stats)):
                        agent.sess.run(agent.update_ops[i], feed_dict={
                            agent.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = agent.sess.run(agent.summary_op)
                    agent.summary_writer.add_summary(summary_str, e + 1)

                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.replay_memory), "  epsilon:", agent.epsilon,
                      "  global_step:", global_step, "  average_q:",
                      agent.avg_q_max / float(step), "  average loss:",
                      agent.avg_loss / float(step))

                agent.avg_q_max, agent.avg_loss = 0, 0

        # 1000 에피소드마다 모델 저장
        if e % 1000 == 0:
            agent.train_model.save_weights("./save_model/Riverraid_dqn.h5")

