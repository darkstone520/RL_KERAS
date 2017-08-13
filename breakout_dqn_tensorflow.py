import numpy as np
import tensorflow as tf
import random
from collections import deque
import dqn
import matplotlib.pyplot as plt
import gym
from typing import List
import psutil, time, os
from skimage.transform import resize
from skimage.color import rgb2gray

env = gym.make('BreakoutDeterministic-v4')
env = gym.wrappers.Monitor(env, directory="gym-results/", force=True)

# Constants defining our neural network
INPUT_DIM = [84,84,4]
OUTPUT_SIZE = 3  # 1, 2, 3
HISTORY_SIZE = 4
DISCOUNT_RATE = 0.99
REPLAY_MEMORY = 600000
TRAIN_START = 50000
BATCH_SIZE = 32
TARGET_UPDATE_FREQUENCY = 5000
MAX_EPISODES = 50000
START_EXPLORATION = 1.0
FINAL_EXPLORATION = 0.1
EXPLORATION = 2000000
E_VALUE = []
EPISODE_LIST = []
TOTAL_REWARD_LIST = []



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

def get_copy_var_ops(*, dest_scope_name: str, src_scope_name: str) -> List[tf.Operation]:

    # Copy variables src_scope to dest_scope
    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder

if __name__ == "__main__":

    # store the previous observations in replay memory
    replay_buffer = deque(maxlen=REPLAY_MEMORY)
    global_step = 0
    last_100_game_reward = deque(maxlen=100)

    with tf.Session() as sess:

        mainDQN = dqn.DQN(sess, INPUT_DIM, OUTPUT_SIZE, name="main", )
        print("mainDQN 생성")
        targetDQN = dqn.DQN(sess, INPUT_DIM, OUTPUT_SIZE, name="target")
        print("targetDQN 생성")
        sess.run(tf.global_variables_initializer())
        e = 1.0
        # initial copy q_net -> target_net
        copy_ops = get_copy_var_ops(dest_scope_name="target",
                                    src_scope_name="main")
        sess.run(copy_ops)

        for episode in range(1,MAX_EPISODES):
            EPISODE_LIST.append(episode)

            done = False
            dead = False
            state = env.reset()
            step = 0
            total_reward, start_life  = 0, 5

            for _ in range(random.randint(1, np.random.randint(1,30))):
                state, _, _, _ = env.step(1)


            state = pre_processing(state)
            history = np.stack((state, state, state, state), axis=2)
            history = np.reshape([history], (1,84,84,4))


            while not done:

                if e > FINAL_EXPLORATION and global_step > TRAIN_START:
                    e -= (START_EXPLORATION - FINAL_EXPLORATION) / EXPLORATION

                if np.random.rand() < e:
                    action = np.random.choice(np.arange(3))
                else:
                    # Choose an action by greedily from the Q-network
                    action = np.argmax(mainDQN.predict(history/255.))

                if action == 0:
                    real_action = 1
                elif action == 1:
                    real_action = 2
                else:
                    real_action = 3

                # Get new state and reward from environment
                next_state, reward, done, info = env.step(real_action)
                global_step += 1
                step += 1


                next_state = pre_processing(next_state)
                next_state = np.reshape([next_state], (1, 84, 84, 1))
                next_history = np.append(next_state, history[:, :, :, :3], axis=3)

                total_reward += reward
                #plot_image(next_history[:,:,:,3])


                mainDQN.avg_q_max += np.amax(
                    mainDQN.predict(np.float32(history / 255.)))


                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']

                reward = np.clip(reward, -1., 1.)

                # Save the experience to our buffer
                replay_buffer.append((history, action, reward, next_history, dead))
                if len(replay_buffer) >= TRAIN_START:
                    minibatch = random.sample(replay_buffer, BATCH_SIZE)
                    mainDQN.replay_train(targetDQN, minibatch)

                    if global_step % TARGET_UPDATE_FREQUENCY == 0:
                        print("TARGET UPDATE")
                        sess.run(copy_ops)

                if dead:
                    dead = False
                else:
                    history = next_history


            if done:
                # 각 에피소드 당 학습 정보를 기록
                if global_step > TRAIN_START:
                    stats = [total_reward, mainDQN.avg_q_max / float(step), step,
                             mainDQN.avg_loss / float(step)]
                    for i in range(len(stats)):
                        mainDQN.sess.run(mainDQN.update_ops[i], feed_dict={
                            mainDQN.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = mainDQN.sess.run(mainDQN.summary_op)
                    mainDQN.summary_writer.add_summary(summary_str, e + 1)

                print("episode:", episode, "  total_reward:", total_reward, "  memory length:",
                      len(replay_buffer), "  epsilon:", e,
                      "  global_step:", global_step, "  average_q:",
                      mainDQN.avg_q_max / float(step), "  average loss:",
                      mainDQN.avg_loss / float(step))

                mainDQN.avg_q_max, mainDQN.avg_loss = 0, 0

        # 1000 에피소드마다 모델 저장
            if episode % 1000 == 0:
                mainDQN.save(episode)
                targetDQN.save(episode)




