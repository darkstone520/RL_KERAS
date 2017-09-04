from model_kkc import Model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
from drawnow import drawnow

def monitorTrainCost(pltSave=False):
    for cost, color, label in zip(mon_cost_list, mon_color_list[0:len(mon_label_list)], mon_label_list):
        plt.plot(mon_epoch_list, cost, c=color, lw=2, ls="--", marker="o", label=label)
    plt.title('Cost Graph per Epoch')
    plt.legend(loc=1)
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.grid(True)
    if pltSave:
        plt.savefig('Cost Graph per Epoch')

def plotImage(image):
    """image array를 plot으로 보여주는 함수
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

def loadInputData():
    """
    텍스트화된 이미지 txt파일을 로드하는 함수
    :return: TRAIN_DATA, TEST_DATA
    """
    print("Loading Data")
    with open(__DATA_PATH + "cat_dog_data", "r", encoding="utf-8") as file:
        # lines : 모든 lines(데이터행)을 불러온다.
        lines = file.readlines()

        # 불러온 전체 lines를 셔플한다.
        lines = random.sample(lines, len(lines))
        lines = random.sample(lines, len(lines))
        lines = random.sample(lines, len(lines))

        # train data를 일정 rate 만큼 뽑아오기 위한 단계
        train_last_index = round(TRAIN_RATE * len(lines))
        file.close()
        # return 시 데이터를 섞어서 return 한다.
        return lines[:train_last_index], lines[train_last_index:]

def loadMiniBatch(lines):
    """
    랜덤 미니배치함수
    txt파일에서 불러온 lines를 읽어서 input data인 numpy array로 바꾸기 위한 함수
    :param lines: loadInputData함수를 통해 불러온 .txt파일의 lines
    :return: numpy arrary의 input data로 X, Y(라벨)을 각각 나누어 리턴한다.
    """
    lines = random.sample(lines, BATCH_SIZE)
    # 각 line은 string으로 되어 있으므로 split한다. split된 리스트 마지막에 '\n'이 추가되므로 [:-1]로 제거한다.
    data = [line.split(',')[:-1] for line in lines]
    data = np.array(data, dtype=np.float32)

    # X,Y를 나누는 작업
    data, label = data[:, :-1], data[:, -1]
    data = data / 255.

    # 라벨을 one_hot으로 바꾼다.
    label = [[1, 0] if label == 0 else [0, 1] for label in label.tolist()]
    label = np.array(label)
    return data, label

def loadBatch(lines, START_BATCH_INDEX):
    """
    일반 배치함수
    :param lines: loadInputData함수를 통해 불러온 .txt파일의 lines
    :param START_BATCH_INDEX: 데이터를 차례로 가져오기 위한 index 값
    :return: numpy arrary의 input data로 X, Y(라벨)을 각각 나누어 리턴한다.
    """
    # 각 line은 string으로 되어 있으므로 split한다. split된 리스트 마지막에 '\n'이 추가되므로 [:-1]로 제거한다.
    lines = lines[START_BATCH_INDEX:START_BATCH_INDEX+BATCH_SIZE]
    START_BATCH_INDEX += BATCH_SIZE
    data = [line.split(',')[:-1] for line in lines]
    data = np.array(data, dtype=np.float32)

    # X,Y를 나누는 작업
    data, label = data[:, :-1], data[:, -1]
    data = data / 255.

    # 고양이, 개 사진이 잘 섞였는지 확인하는 부분
    # for idx, l in enumerate(label):
    #     if l == 0:
    #         print(l)
    #         print("고양이입니다.")
    #     else:
    #         print(l)
    #         print("개입니다.")
    #     plot_image(data[idx].reshape(144,144))

    # 라벨을 one_hot으로 바꾼다.
    label = [[1, 0] if label == 0 else [0, 1] for label in label.tolist()]
    label = np.array(label)
    return data, label

# early stopping하기 위해 테스트 하는 것을 별도 함수로 구현
def validateModel(MODEL_ACCURACY):

    START_BATCH_INDEX = 0
    ENSEMBLE_ACCURACY = 0
    CNT = 0

    with tf.Session() as sess:

        print('Test Start!')
        models = []
        for m in range(NUM_MODELS):
            models.append(Model(sess, "model" + str(m)))

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, 'log/epoch_' + str(LAST_EPOCH) + '.ckpt')

        for _ in range(TEST_EPHOCHS):

            # 총 데이터의 갯수가 배치사이즈로 나누어지지 않을 경우 버림한다
            total_batch_num = math.trunc(len(TEST_DATA) / BATCH_SIZE)

            for i in range(total_batch_num):

                print("Test Batch Data Reading {}/{}".format(i + 1, total_batch_num))

                # test_x_batch, test_y_batch = loadMiniBatch(TEST_DATA)
                test_x_batch, test_y_batch = loadBatch(TEST_DATA, START_BATCH_INDEX)

                test_size = len(test_y_batch) # 테스트 데이터
                predictions = np.zeros(test_size * 2).reshape(test_size, 2) # [[0.0, 0.0], [0.0, 0.0] ...]
                model_result = np.zeros(test_size * 2, dtype=np.int).reshape(test_size, 2)  #[ [0,0], [0,0]...]
                model_result[:, 0] = range(0, test_size) # [[0,0],[1,0], [2,0], [3,0] ......]

                for idx, m in enumerate(models):
                    MODEL_ACCURACY[idx] += m.get_accuracy(test_x_batch, test_y_batch) # 모델의 정확도가 각 인덱스에 들어감 [0.92, 0.82, 0.91]
                    p = m.predict(test_x_batch) # 모델이 분류한 라벨 값
                    model_result[:, 1] = np.argmax(p, 1) #  두번째 인덱스에 p중 가장 큰값을 넣는다 [[0,0],[1,1], [2,1], [3,0] ......]
                    for result in model_result:
                        predictions[result[0], result[1]] += 1

                ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(test_y_batch, 1))
                ENSEMBLE_ACCURACY += tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
                CNT += 1

            START_BATCH_INDEX = 0

        for i in range(len(MODEL_ACCURACY)):
            print('Model ' + str(i) + ' : ', MODEL_ACCURACY[i] / CNT)
        print('Ensemble Accuracy : ', sess.run(ENSEMBLE_ACCURACY) / CNT)
        print('Testing Finished!')
        return ENSEMBLE_ACCURACY


# 학습을 위한 기본적인 셋팅
__DATA_PATH = "preprocessed_data/"
IMG_SIZE = (144, 144)
BATCH_SIZE = 100
START_BATCH_INDEX = 0
TRAIN_EPOCHS = 14
TEST_EPHOCHS = 1
TRAIN_RATE = 0.8
NUM_MODELS = 3
MINI_BATCH = True
MODEL_ACCURACY = np.zeros(NUM_MODELS).tolist()
LAST_EPOCH = None

# monitoring 관련 parameter
mon_epoch_list = []
mon_cost_list = [[] for m in range(NUM_MODELS)]
mon_color_list = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
mon_label_list = ['model'+str(m+1) for m in range(NUM_MODELS)]

# TRAIN_DATA와 TEST_DATA를 셋팅, 실제 각 변수에는 txt파일의 각 line 별 주소 값이 리스트로 담긴다.
TRAIN_DATA, TEST_DATA = loadInputData()

# TRAIN
with tf.Session() as sess:
    # 시작 시간 체크
    stime = time.time()
    models = []
    # initialize
    for m in range(NUM_MODELS):
        models.append(Model(sess, "model" + str(m)))

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    print('Learning Started!')

    # train my model
    for epoch in range(TRAIN_EPOCHS):

        avg_cost_list = np.zeros(len(models))

        # 총 데이터의 갯수가 배치사이즈로 나누어지지 않을 경우 버림한다
        total_batch_num = math.trunc(int(len(TRAIN_DATA) / BATCH_SIZE))

        for i in range(total_batch_num):

            print("{} Epoch: Batch Data Reading {}/{}".format(epoch+1, i + 1, total_batch_num))

            # MINI_BATCH 여부에 따라 나뉜다.
            if MINI_BATCH:
                if epoch < 2:
                    train_x_batch, train_y_batch = loadBatch(TRAIN_DATA,START_BATCH_INDEX)
                else:
                    train_x_batch, train_y_batch = loadMiniBatch(TRAIN_DATA)
            else:
                train_x_batch, train_y_batch = loadBatch(TRAIN_DATA, START_BATCH_INDEX)

            # train each model
            for m_idx, m in enumerate(models):
                c, _ = m.train(train_x_batch, train_y_batch)
                avg_cost_list[m_idx] += c / total_batch_num

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost_list)
        START_BATCH_INDEX = 0
        LAST_EPOCH = epoch+1

        mon_epoch_list.append(epoch + 1)
        for idx, cost in enumerate(avg_cost_list):
            mon_cost_list[idx].append(cost)
        drawnow(monitorTrainCost)

    drawnow(monitorTrainCost, pltSave=True)
    print('Learning Finished!')
    saver.save(sess, 'log/epoch_' + str(LAST_EPOCH) + '.ckpt')

    # 종료 시간 체크
    etime = time.time()
    print('consumption time : ', round(etime - stime, 6))


tf.reset_default_graph()
validateModel(MODEL_ACCURACY)



