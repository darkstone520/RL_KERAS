import numpy as np
import tensorflow as tf
import random
from collections import deque
import dqn_BN
import matplotlib.pyplot as plt
import gym
from typing import List
from skimage.transform import resize
from skimage.color import rgb2gray


# gym 게임 설정, monitor 동영상 촬영 설정
env = gym.make('RiverraidDeterministic-v4')
env = gym.wrappers.Monitor(env, directory="gym-results/", force=True)


INPUT_DIM = [84,84,4]            # 신경망 INPUT DIM으로 frame 사이즈 (84,84) 4개(HISTORY_SIZE)를 1개의 state로 사용
OUTPUT_SIZE = 7                  # 각 게임의 Action 갯수
HISTORY_SIZE = 4                 # 신경망에 frame 화면을 몇 개씩 보여줄 것인지 설정
REPLAY_MEMORY = 400000           # 게임을 진행하면서 샘플데이터(history, action, reward, next_history, dead)를 쌓아 놓을 que의 길이
TRAIN_START = 50000              # 샘플데이터가 몇개 쌓였을 때 Train을 시작할 지 정하는 변수
BATCH_SIZE = 32                  # 미니배치에서 사용할 배치사이즈 설정
TARGET_UPDATE_FREQUENCY = 15000  # 몇 Frame 마다 Target 신경망을 업데이트 할 지 정하는 변수
MAX_EPISODES = 50000             # 게임을 플레이 할 최대 EPISODE
START_EXPLORATION = 1.0          # Epsilon 시작 값 (Exploration and Exploit Greedy 설정 관련 변수)
FINAL_EXPLORATION = 0.1          # Epsilon 마지막 값 (Exploration and Exploit Greedy 설정 관련 변수)
EXPLORATION = 2000000            # Epsilon 시작부터 마지막 값까지 몇개의 step(time_step, frame, action)으로 줄여나갈 지 정하는 변수



def plot_image(image):
    """게임 화면을 plot으로 보여주는 함수
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

def pre_processing(observe):
    observe = crop_image(observe)
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe

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

# MainDQN의 weight 값을 TargetDQN에 복사하는 함수
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

    # 학습을 위한 경험데이터 que 변수
    replay_buffer = deque(maxlen=REPLAY_MEMORY)
    global_step = 0

    with tf.Session() as sess:

        mainDQN = dqn_BN.DQN(sess, INPUT_DIM, OUTPUT_SIZE, name="main", )
        print("mainDQN 생성")
        targetDQN = dqn_BN.DQN(sess, INPUT_DIM, OUTPUT_SIZE, name="target")
        print("targetDQN 생성")
        sess.run(tf.global_variables_initializer())
        e = 1.0

        # MainDQN과 TargetDQN을 게임 시작 시 동일하게 복사함
        copy_ops = get_copy_var_ops(dest_scope_name="target",
                                    src_scope_name="main")
        sess.run(copy_ops)

        # Episode 시작
        for episode in range(MAX_EPISODES):

            done = False            # 게임 종료 여부
            dead = False            # 죽었는지 여부
            state = env.reset()     # 게임 시작을 위해 env 환경 초기화
            state , _, _, info = env.step(1)
            step, total_reward, start_life = 0, 0, info['ale.lives']


            # (210,160,3)의 raw image를 전처리
            state = pre_processing(state)

            # 전처리된 화면을 4개로 하는 history(신경망이 바라볼 화면)을 만들어 줌
            history = np.stack((state, state, state, state), axis=2)  #shape = (84,84,4)
            history = np.reshape([history], (1,84,84,4))              # 신경망에서는 4차원 input을 사용하기 때문에 차원을 늘려줌

            # EPSIODE 1번, 즉 가진 목숨을 모두 소진할 때 까지 Play
            while not done:

                # Expliot and Exploration Greedy 를 위한 Epsilon 값을 감소하는 부분
                # Train Start Frame이 될 때까지(학습데이터가 50000개 쌓일 때까지) Epsilon 값을 감소하지 않음
                if e > FINAL_EXPLORATION and global_step > TRAIN_START:
                    e -= (START_EXPLORATION - FINAL_EXPLORATION) / EXPLORATION

                # Epsilon이 1보다 작아지는 순간부터 랜덤의 확률로 random action 또는 모델 predict action을 취함
                if np.random.rand() < e:
                    action = np.random.choice(np.arange(7))
                else:
                    # Choose an action by greedily from the Q-network
                    action = np.argmax(mainDQN.predict(history/255.))

                # 위의 action은 index를 출력하기 때문에 각 index 별로 실제 action을 매칭해줌
                # 1: 전진 + 발사, 3: 오른쪽, 4: 왼쪽, 5: 전진 천천히
                # 12: 왼쪽대각선전진+발사, 13: 전진(천천히) + 발사,
                # 14: 오른쪽대각선전진+발사
                if action == 0:
                    real_action = 1
                elif action == 1:
                    real_action = 3
                elif action == 2:
                    real_action = 4
                elif action == 3:
                    real_action = 5
                elif action == 4:
                    real_action = 12
                elif action == 5:
                    real_action = 13
                elif action == 6:
                    real_action = 14


                # 위에서 발생한 action을 게임환경에서 움직이게 하고 반환된 state, reward, done, info 값을 저장함
                next_state, reward, done, info = env.step(action)
                global_step += 1           # action 한번에 step 한번
                step += 1                  # action 한번에 step 한번


                next_state = pre_processing(next_state)                            # raw image data를 다시한번 전처리
                next_state = np.reshape([next_state], (1, 84, 84, 1))              # hisotry로 저장하기 위해 shape 변환
                next_history = np.append(next_state, history[:, :, :, :3], axis=3) # new frame이 old frame을 밀어냄

                total_reward += reward
                #plot_image(next_history[:,:,:,3])


                # Tensorboard를 사용하기 위해 값을 저장
                mainDQN.avg_q_max += np.amax(
                    mainDQN.predict(np.float32(history / 255.)))


                # 목숨을 1개 잃어 을 때 Dead처리 해주어 Train 할 때 죽었을 때 reward에 대한 업데이트를 하지 못하도록 방지하기 위함
                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']
                    reward = -100

                # 목숨이 1개밖에 없는 게임의 경우 done이 되었을 때 dead 처리하기 위함
                if done:
                    dead = True
                    reward = -100

                # DQN에서 train 시 reward를 정규화할 것이기 때문에 clip 해줄 필요가 없음
                #reward = np.clip(reward, -1., 1.)


                # 경험 리플레이 메모리에 데이터를 쌓음
                replay_buffer.append((history, action, reward, next_history, dead))
                if len(replay_buffer) >= TRAIN_START:
                    minibatch = random.sample(replay_buffer, BATCH_SIZE)
                    mainDQN.replay_train(targetDQN, minibatch)

                    if global_step % TARGET_UPDATE_FREQUENCY == 0:
                        print("TARGET UPDATE")
                        sess.run(copy_ops)

                # 목숨만 잃고 게임을 끝나지 않았을 경우 dead를 다시 False로 바꿔주어야 함
                if dead:
                    dead = False

                # 목숨을 잃지 않은 경우 계속해서 게임 진행
                else:
                    history = next_history


            # 텐서보드를 위한 설정
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

            # 1000판마다 모델 저장
            if episode % 1000 == 0:
                mainDQN.save(episode)
                targetDQN.save(episode)




