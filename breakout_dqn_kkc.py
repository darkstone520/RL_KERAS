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

EPISODES = 50000


# 브레이크아웃에서의 DQN 에이전트
class DQNAgent:
    def __init__(self, action_size):
        self.render = False
        self.load_model = False
        # 상태와 행동의 크기 정의
        self.state_size = (84, 84, 4)
        self.action_size = action_size
        # DQN 하이퍼파라미터
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps
        self.batch_size = 32
        self.train_start = 50000
        self.update_target_rate = 10000
        self.discount_factor = 0.99
        # 리플레이 메모리, 최대 크기 400000
        self.memory = deque(maxlen=400000)
        self.no_op_steps = 30
        # 모델과 타겟모델을 생성하고 타겟모델 초기화
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        self.optimizer = self.optimizer()

        # 텐서보드 설정
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            'summary/breakout_dqn', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        if self.load_model:
            self.model.load_weights("./save_model/breakout_dqn.h5")

    # Huber Loss를 이용하기 위해 최적화 함수를 직접 정의
    def optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        # history에서 예측한 q value
        prediction = self.model.output

        # history에서 취한 action을 one_hot
        # action이 0일 때 [1,0,0]
        a_one_hot = K.one_hot(a, self.action_size)

        # [1,0,0] * [0.32113, 0.1123, 0.00123] = [0.32113,0,0]과 같이 된다.
        # 위의 sum하면 0.32113
        q_value = K.sum(prediction * a_one_hot, axis=1)
        # target(reward + dicount_factor * np.max(model.predict(next_history)) )으로 받은 값이 y
        # 즉 정답인 y - q_value만큼의 오차가 발생하는데 현재 q_value의 값이 잘 못 되었을 경우 오차가 크게 발생
        # 위에서 q_value는 우리가 예측한 값이다. (q hat)
        error = K.abs(y - q_value)


        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        optimizer = RMSprop(lr=0.00025, epsilon=0.01)
        # updates는 list
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates=updates)
        """Instantiates a Keras function.

            # Arguments
                inputs: List of placeholder tensors.
                outputs: List of output tensors.
                updates: List of update ops.
                **kwargs: Passed to `tf.Session.run`.

            # Returns
                Output values as Numpy arrays.

            # Raises
                ValueError: if invalid kwargs are passed in.
            """
        return train

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_size, kernel_initializer='he_normal'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu', kernel_initializer='he_normal'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', kernel_initializer='he_normal'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(self.action_size))
        model.summary()
        return model

    # 타겟 모델을 모델의 가중치로 업데이트
    # 타겟 모델을 만드는 이유 -> 학습과 동시에 target값으 구하면 target model 자체가 수시로 흔들리기 때문에 학습이 되지 않는다. (non-stationary)
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    # 랜덤 미니배치를 쓰는 이유 -> sample data들의 correlations를 제거할 수 있음
    def train_model(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        mini_batch = random.sample(self.memory, self.batch_size)

        # shape = (batch_size, 84, 84, 4)
        # agent.append_sample(history, action, reward, next_history, dead)
        history = np.vstack([x[0]/255. for x in mini_batch])
        action = np.array([x[1] for x in mini_batch])
        reward = np.array([x[2] for x in mini_batch])
        next_history = np.vstack([x[3]/255. for x in mini_batch])
        dead = np.array([x[4] for x in mini_batch])

        target = reward + self.discount_factor * np.max(self.target_model.predict(next_history),axis=-1) * ~dead

        #print(reward)
        #print(np.max(self.model.predict(history),axis=-1))
        #print(dead)
        #print(target)

        # reward
        #                                                13번째
        # [0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.
        #  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]

        # 모델이 예측한  max Q value * 0.99 곱한 값
        # [0.10333048  0.13567154  0.11365376  0.10544886  0.11376774  0.10947491
        #  0.09684324  0.10063805  0.11025954  0.11438356  0.11261801  0.1120755
        #  <0.09465584>  0.137052    0.11423227  0.10483754  0.13442372  0.10209396
        #  0.1023718   0.10295425  0.11089235  0.11386036  0.10653776  0.0947252
        #  0.10454454  0.10452821  0.10982917  0.09662293  0.09869094  0.10803009
        #  0.11695433  0.1101466]

        # 게임 종료 여부
        # [False False False False False False False False False False False False
        #  False False False False False False False False False False False False
        #  False False False False False False False False]

        # target 값
        # [0.10333048  0.13567154  0.11365376  0.10544886  0.11376774  0.10947491
        #  0.09684324  0.10063805  0.11025954  0.11438356  0.11261801  0.1120755
        #  <1.09465584>  0.137052    0.11423227  0.10483754  0.13442372  0.10209396

        # loss = (정답-예측)^2
        loss = self.optimizer([history, action, target])
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


# 학습속도를 높이기 위해 흑백화면으로 전처리
# 흑백으로 바꾼 후 255를 곱해서 연산이 빠르도록 int값으로 바꿈
def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe


def plot_image(image):
    """Plot an image

    If an image is a grayscale image,
    plot in `gray` cmap.
    Otherwise, regular RGB plot
    .

    Args:
        image (2-D or 3-D array): (H, W) or (H, W, C)
    """
    image = np.squeeze(image)
    shape = image.shape

    if len(shape) == 2:
        plt.imshow(image, cmap="gray")

    else:
        plt.imshow(image)

    plt.show()

if __name__ == "__main__":
    # 환경과 DQN 에이전트 생성
    env = gym.make('BreakoutDeterministic-v4')
    env = gym.wrappers.Monitor(env, directory="gym-results/", force=True)
    agent = DQNAgent(action_size=3)

    scores, episodes, global_step = [], [], 0

    for e in range(EPISODES):
        done = False
        dead = False

        step, score, start_life = 0, 0, 5
        observe = env.reset()

        for _ in range(random.randint(1, agent.no_op_steps)):
            observe, _, _, _ = env.step(1)

        state = pre_processing(observe)
        # state, shape = (84,84)
        # [[0 0 0..., 0 0 0]
        #  [0 0 0..., 0 0 0]
        #  ...
        #  [0 0 0..., 0 0 0]
        #  [0 0 0..., 0 0 0]]

        # history, shape = (84,84,4), 4에 해당하는 축으로 이미지를 쌓아야 하므로 axis=2가된다. 또는 가장 안쪽의 축 -1
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))
        while not done:
            if agent.render:
                env.render()
            global_step += 1
            step += 1

            # 바로 전 4개의 상태로 행동을 선택
            action = agent.get_action(history)
            # 1: 정지, 2: 왼쪽, 3: 오른쪽
            if action == 0:
                real_action = 1
            elif action == 1:
                real_action = 2
            else:
                real_action = 3

            # 선택한 행동으로 환경에서 한 타임스텝 진행
            observe, reward, done, info = env.step(real_action)
            # 각 타임스텝마다 상태 전처리
            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            # netxt_state를 history[:,:,:,0] 번째 인덱스에 넣게 됨
            # 가장 최신의 state를 history axis=3의 index 0부터 시작하게 한다.
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            agent.avg_q_max += np.amax(
                agent.model.predict(np.float32(history/255.))[0])

            # 가장 최근의 저장된 state를 ( history[:,:,:,0] ) 확인
            #plot_image(next_history[:,:,:,0])

            # print(agent.model.predict(np.float32(history / 255.)))  # [[-0.03298247  0.01585374 -0.00639954]]
            # print(np.amax(agent.model.predict(np.float32(history / 255.))[0])) # 0.0164134

            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            # 1이 넘는 reward도 1로 취급 (보상의 차이가 크면 학습이 어려움)
            reward = np.clip(reward, -1., 1.)
            # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장 후 학습
            agent.append_sample(history, action, reward, next_history, dead)

            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            # 일정 시간마다 타겟모델을 모델의 가중치로 업데이트
            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()

            score += reward

            if dead:
                dead = False
            else:
                history = next_history

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
                      len(agent.memory), "  epsilon:", agent.epsilon,
                      "  global_step:", global_step, "  average_q:",
                      agent.avg_q_max / float(step), "  average loss:",
                      agent.avg_loss / float(step))

                agent.avg_q_max, agent.avg_loss = 0, 0

        # 1000 에피소드마다 모델 저장
        if e % 1000 == 0:
            agent.model.save_weights("./save_model/breakout_dqn.h5")
