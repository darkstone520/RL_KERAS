from resnet_no_bottle_26layers import Model as Model_26
from resnet_no_bottle_18layers import Model as Model_18
from resnet_BNK_50layers import Model as My
# from model_kkc_hanwha import Model as My
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
from scipy import ndimage
from drawnow import drawnow
from pandas_ml import ConfusionMatrix
from collections import deque
from PIL import Image
import scipy.misc as sc
import os

def monitorAccuracy(epoch_num, pltSave=False):

    plt.figure(1)
    for cost, color, label in zip(mon_cost_list, mon_color_list[0:len(mon_label_list_for_cost)], mon_label_list_for_cost):
        if epoch_num == 1:
            plt.plot(mon_epoch_list, cost, c=color, lw=2, ls="-", marker="None", label=label)
        else:
            plt.plot(mon_epoch_list, cost, c=color, lw=2, ls="-", marker="None", label="_nolegend_")

    plt.title('Cost Graph per Epoch')
    plt.legend(loc=1)
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.grid(True)
    if pltSave:
        plt.savefig('Cost Graph per Epoch {}_{}'.format(CLASS_NUM,time.asctime()))

    plt.figure(2)
    for accuracy, color, label in zip(mon_acuuracy_list, mon_color_list[0:len(mon_label_list)], mon_label_list):
        plt.plot(mon_epoch_list, accuracy, c=color, lw=2, ls="-", marker="None", label=label)
    plt.title('Error Graph per Epoch')
    plt.legend(loc=1)
    plt.xlabel('Epoch')
    plt.ylabel('Error %')
    plt.grid(True)
    if pltSave:
        plt.savefig('Error Graph per Epoch {}_{}'.format(CLASS_NUM,time.asctime()))

def ImageSave(image, filename):
    """image array를 plot으로 보여주는 함수
    Args:
        image (2-D or 3-D array): (H, W) or (H, W, C)
    """
    image = np.squeeze(image)
    shape = image.shape
    fig = plt.gcf()
    if len(shape) == 2:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)
    fig.savefig(filename +'.jpg')



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

def loadTrainData():
    """
    텍스트화된 이미지 txt파일을 로드하는 함수
    :return: TRAIN_DATA, TEST_DATA
    """
    print("Loading Data")
    with open(__DATA_PATH + "car_door_norm_scr_broken_train_3000", "r", encoding="utf-8") as file:
        # lines : 모든 lines(데이터행)을 불러온다.
        lines = file.readlines()

        # 불러온 전체 lines를 셔플한다.
        lines = random.sample(lines, len(lines))
        lines = random.sample(lines, len(lines))
        lines = random.sample(lines, len(lines))

        # train data를 일정 rate 만큼 뽑아오기 위한 단계
        train_last_index = round(TRAIN_RATE * len(lines))

        file.close()

        # 테스트용 리턴값
        # return lines[:2], lines[2:4]

        # return 시 데이터를 섞어서 return 한다.
        return lines

def loadTestData():
    """
    텍스트화된 이미지 txt파일을 로드하는 함수
    :return: TRAIN_DATA, TEST_DATA
    """
    print("Loading Data")
    with open(__DATA_PATH + "car_door_norm_scr_broken_test_120", "r", encoding="utf-8") as file:
        # lines : 모든 lines(데이터행)을 불러온다.
        lines = file.readlines()

        # 불러온 전체 lines를 셔플한다.
        lines = random.sample(lines, len(lines))
        lines = random.sample(lines, len(lines))
        lines = random.sample(lines, len(lines))

        # train data를 일정 rate 만큼 뽑아오기 위한 단계
        train_last_index = round(TRAIN_RATE * len(lines))

        file.close()

        # 테스트용 리턴값
        # return lines[:2], lines[2:4]

        # return 시 데이터를 섞어서 return 한다.
        return lines

def loadRandomMiniBatch(lines):
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


    label_list = []

    for label in label.tolist():
        if label == 0:
            label_list.append([1,0,0])
        elif label == 1:
            label_list.append([0,1,0])
        elif label == 2:
            label_list.append([0,0,1])


    label = np.array(label_list)
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


    label_list = []

    for label in label.tolist():
        if label == 0:
            label_list.append([1,0,0])
        elif label == 1:
            label_list.append([0,1,0])
        elif label == 2:
            label_list.append([0,0,1])


    label = np.array(label_list)
    return data, label, START_BATCH_INDEX

def onehot2label(label_array):

    label_list = np.argmax(label_array, axis=1)
    return label_list.tolist()

def shuffleLines(lines):
    lines = random.sample(lines, len(lines))
    return random.sample(lines, len(lines))

def predictConsumtionTime(epoch_num):
    """
    총소요시간 = 배치 사이즈 * a * 앙상블 모델의 수 * 배치 횟수(per Epoch) * 전체 Ephoc 수
    0.0053 : 이미지 1개당 연산 시간(현재모델 0.0053)
    alpha = 총소요시간 / (배치사이즈 * 앙상블 모델의 수 * 배치 횟수 * 전체 Ephoc 수)
    alpha 를 구하기 위해서는 전체 소요시간을 1회 측정해서 구해야한다.
    """
    alpha = 0.0053
    c_time = BATCH_SIZE * 0.0053 * NUM_MODELS * math.trunc(int(len(TRAIN_DATA)/BATCH_SIZE)) * epoch_num
    c_time = float(c_time/60)
    print("총 {} 에폭 학습, 학습 예상 소요시간: {} 분".format(epoch_num,c_time))
    return c_time

def distortImage(images):
    return ndimage.uniform_filter(images, size=11)

def randomCrop(image_array, multi_scaling=True):
    if multi_scaling:
        scale_range = random.sample([32,160,288], k=1)[0]
    else:
        scale_range = 32
    origin_size = image_array.shape
    rnd_width = random.randint(0,scale_range)
    rnd_height = random.randint(0,scale_range)
    image_array = np.pad(image_array, (scale_range/2,scale_range/2), "constant")

    # Image Crop 단계
    image_array = image_array[
                  # width
                  rnd_width:origin_size[0]+rnd_width,
                  # height
                  rnd_height:origin_size[1]+rnd_height
                  ]
    return image_array



# 학습을 위한 기본적인 셋팅
__DATA_PATH = "preprocessed_data/"
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
START_BATCH_INDEX = 0

# 학습 도중 이미지를 Distort하는 데이터의 비중
IMAGE_DISTORT_RATE = 0

# EARLY_STOP 시작하는 에폭 시점
START_EARLY_STOP_EPOCH = 1
START_EARLY_STOP_COST = 10

TRAIN_RATE = 1.0
NUM_MODELS = 3
CLASS_NUM = 3
TEST_ACCURACY_LIST = []
START_BATCH_INDEX = 0


# Random Mini Batch의 데이터 중복 허용 여부를 정한다. 순서(Order)가 True 경우 중복이 허용되지 않는다.
# 둘다 False 일 경우 : Random mini batch no order(데이터 중복허용)을 수행

RANDOM_MINI_BATCH_NO_ORDER = True # 중복
MIN_ORDER_BATCH_EPCHO = 0 # Random mini batch 시 Normal Batch를 몇 회 수행 후 미니배치를 수행할 것인지 정하는 변수

RANDOM_MINI_BATCH_ORDER = False # 중복없는 랜덤 미니배치
NORMAL_BATCH = False # 일반배치
LAST_EPOCH = None

################################
## monitoring 관련 parameter
################################
mon_epoch_list = []
# mon_color_list = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
mon_color_list = ['blue','cyan', 'magenta', 'gold']
mon_label_list_for_cost = ['My Model','ResNet_18','ResNet_26']
mon_label_list = ['My Model','ResNet_18', 'ResNet_26']
# mon_label_list_for_cost = ['model'+str(m+1) for m in range(NUM_MODELS)]
# mon_label_list = ['model'+str(m+1) for m in range(NUM_MODELS)]
# cost monitoring 관련
mon_cost_list = [[] for m in range(NUM_MODELS)]
# accuracy monitoring 관련
mon_acuuracy_list = [[] for m in range(NUM_MODELS+1)]


# TRAIN_DATA와 TEST_DATA를 셋팅, 실제 각 변수에는 txt파일의 각 line 별 주소 값이 리스트로 담긴다.
stime = time.time()
TRAIN_DATA = loadTrainData()
TEST_DATA = loadTestData()
#loadAllTestLabel(TEST_DATA)

print("Train Data {}개 , Test Data {}개 ".format(len(TRAIN_DATA), len(TEST_DATA)))


# 종료 시간 체크
etime = time.time()
print('Data Loading Consumption Time : ', round(etime - stime, 6))
predictConsumtionTime(START_EARLY_STOP_EPOCH)

# TRAIN
with tf.Session() as sess:
    # 시작 시간 체크
    stime = time.time()
    models = []
    valid_result = []
    epoch = 0
    temp = deque(maxlen=2)
    # initialize

    # initialize
    # for m in range(NUM_MODELS):
    #     models.append(Model(sess, "model" + str(m), CLASS_NUM))

    # for m in range(1):
    #     models.append(My(sess, "My_Model", CLASS_NUM))
    # for m in range(1):
    #     models.append(Model_18(sess, "ResNet_18", CLASS_NUM))
    for m in range(3):
        models.append(My(sess, "My_" + str(m), CLASS_NUM))

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    print('Learning Started!')

    # train my model
    # for epoch in range(TRAIN_EPOCHS):
    while True:
        avg_cost_list = np.zeros(len(models))
        avg_accuracy_list = np.zeros(len(models)+1)

        # 총 데이터의 갯수가 배치사이즈로 나누어지지 않을 경우 버림한다
        total_batch_num = math.trunc(int(len(TRAIN_DATA) / BATCH_SIZE))

        ################################################################################
        ###  - 랜덤 미니배치(데이터 중복 또는 중복 불가) 또는 일반배치를 설정하는 부분
        ###  - RANDOM_MINI_BATCH_ORDER의 Boolen 값에 따라 수행하는 것이 달라진다.
        ################################################################################

        # 랜덤 미니배치 중복없이 할 경우 매 에폭마다 Train Data를 섞어준다.
        if RANDOM_MINI_BATCH_ORDER:
            # print("랜덤 미니배치(중복불가)를 수행합니다. Data Shuffle")
            TRAIN_DATA = shuffleLines(TRAIN_DATA)
        # elif NORMAL_BATCH:
        #     print("일반 배치(중복불가)를 수행합니다.")
        # else:
        #     print("랜덤 미니배치(중복허용)를 수행합니다.")

        for i in range(total_batch_num):

            # MINI_BATCH 여부에 따라 나뉜다.
            # 중복 없는 Random Mini Batch
            if RANDOM_MINI_BATCH_ORDER:
                # print("[데이터 중복 불가] {} Epoch: Random Mini Batch Data Reading {}/{}, DATA INDEX : {}".
                #       format(epoch + 1, i + 1, total_batch_num,START_BATCH_INDEX))
                train_x_batch, train_y_batch, START_BATCH_INDEX = loadBatch(TRAIN_DATA, START_BATCH_INDEX)

            # Normal Batch
            elif NORMAL_BATCH:
                # print("[데이터 중복 불가] {} Epoch: Normal Batch Data Reading {}/{}, DATA INDEX : {}".
                #       format(epoch + 1, i + 1, total_batch_num, START_BATCH_INDEX))
                train_x_batch, train_y_batch, START_BATCH_INDEX = loadBatch(TRAIN_DATA, START_BATCH_INDEX)

            # 중복 허용 Random Mini Batch
            elif RANDOM_MINI_BATCH_NO_ORDER:

                # 특정 Epoch만큼 데이터 중복없이 일반배치 또는 랜덤미니배치를 수행을 설정하는 부분
                if epoch < MIN_ORDER_BATCH_EPCHO:
                    # print("[데이터 중복 불가] {}/{} Epoch : Normal Batch Data Reading {}/{}, DATA INDEX : {}".
                    #       format(epoch + 1, MIN_ORDER_BATCH_EPCHO, i + 1, total_batch_num, START_BATCH_INDEX))
                    train_x_batch, train_y_batch, START_BATCH_INDEX = loadBatch(TRAIN_DATA,START_BATCH_INDEX)
                else:
                    # print("[데이터 중복 허용] {} Epoch: Random Mini Batch Data Reading {}/{}".
                    #       format(epoch + 1, i + 1, total_batch_num))
                    train_x_batch, train_y_batch = loadRandomMiniBatch(TRAIN_DATA)


            # crop data augmentation
            cropped_train_x_batch = []
            for i in train_x_batch:
                image = i.reshape(224,224)
                image = randomCrop(image)
                cropped_train_x_batch.append(image.flatten())
            train_x_batch = np.array(cropped_train_x_batch).reshape(-1,224*224)

            # 이미지 왜곡
            # if IMAGE_DISTORT_RATE > random.random():
            #     train_x_batch = distortImage(train_x_batch)

            # Train each model
            for m_idx, m in enumerate(models):
                c, _ = m.train(train_x_batch, train_y_batch)
                a    = m.get_accuracy(train_x_batch, train_y_batch)
                avg_accuracy_list[m_idx] += a / total_batch_num
                avg_cost_list[m_idx] += c / total_batch_num

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost_list)
        START_BATCH_INDEX = 0
        LAST_EPOCH = epoch+1

        mon_epoch_list.append(epoch + 1)
        for idx, cost in enumerate(avg_cost_list):
            mon_cost_list[idx].append(cost)
        for idx, accuracy in enumerate(avg_accuracy_list):
            if idx != len(avg_accuracy_list)-1:
                mon_acuuracy_list[idx].append(round((1.0-accuracy)*100,3))
        # drawnow(monitorTrainCost)

        epoch += 1

        ###################################################################################
        ## Early Stop, Test 검증
        ################################################################################
        if (epoch >= START_EARLY_STOP_EPOCH) and float(np.mean(avg_cost_list)) < START_EARLY_STOP_COST:
            if epoch == 1:
                mon_label_list.append("Test")
            # Test 수행 시 마다 초기화가 필요한 변수들
            MODEL_ACCURACY = np.zeros(NUM_MODELS).tolist()
            CNT = 0
            TEST_ACCURACY = None
            ENSEMBLE_ACCURACY = 0
            TEST_DATA = shuffleLines(TEST_DATA)
            test_total_batch_num = math.trunc(len(TEST_DATA) / BATCH_SIZE)
            ALL_TEST_LABELS = []
            predictions = np.zeros(test_total_batch_num * BATCH_SIZE * CLASS_NUM)\
                .reshape(-1,CLASS_NUM)  # [[0.0, 0.0], [0.0, 0.0] ...]
            softmax_predictions = np.zeros(test_total_batch_num * BATCH_SIZE * CLASS_NUM)\
                .reshape(-1,CLASS_NUM)  # [[0.0, 0.0], [0.0, 0.0] ...]

            # print("{} Epoch 모델에 대한 검증을 시작합니다.".format(epoch))
            # 모델 검증
            # 총 데이터의 갯수가 배치사이즈로 나누어지지 않을 경우 버림한다
            for i in range(test_total_batch_num):

                # print("Test Batch Data Reading {}/{}, DATA INDEX : {}".format(i + 1, test_total_batch_num, START_BATCH_INDEX))
                # test_x_batch, test_y_batch = loadMiniBatch(TEST_DATA)
                test_x_batch, test_y_batch, START_BATCH_INDEX = loadBatch(TEST_DATA, START_BATCH_INDEX) # 리턴 시 START_BATCH_INDEX는 + BATCH_SZIE 되어 있음
                ALL_TEST_LABELS.append(test_y_batch)

                # 모든 앙상블 모델들에 대해 각각 모델의 정확도와 predict를 구하는 과정
                for idx, m in enumerate(models):
                    MODEL_ACCURACY[idx] += m.get_accuracy(test_x_batch,
                                                          test_y_batch)  # 모델의 정확도가 각 인덱스에 들어감 [0.92, 0.82, 0.91]
                    p = m.predict(test_x_batch)  # 모델이 분류한 라벨 값
                    # 위에서 load배치 함수를 호출하면 START_BATCH_INDEX가 BATCH_SIZE만큼 증가하기 때문에 다시 빼준다.
                    predictions[START_BATCH_INDEX-BATCH_SIZE:START_BATCH_INDEX,:] += p
                    s_p = m.predict_softmax(test_x_batch)
                    softmax_predictions[START_BATCH_INDEX-BATCH_SIZE:START_BATCH_INDEX,:] += s_p


                CNT += 1
            ALL_TEST_LABELS = np.array(ALL_TEST_LABELS).reshape(-1,CLASS_NUM)
            # softmax값이 앙상블 모델별로 누적되어있으니깐 모델갯수로 나누어 평균을 구한다.
            softmax_predictions = softmax_predictions/NUM_MODELS
            ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(ALL_TEST_LABELS, 1))

            ENSEMBLE_ACCURACY += tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))

            START_BATCH_INDEX = 0

            for i in range(len(MODEL_ACCURACY)):
                print('Model ' + str(i) + ' : ', MODEL_ACCURACY[i] / CNT)
            TEST_ACCURACY = sess.run(ENSEMBLE_ACCURACY)
            temp.append(TEST_ACCURACY)
            # print(TEST_ACCURACY)
            print('Ensemble Accuracy : ', TEST_ACCURACY)


            if TEST_ACCURACY > 0.5:
                # 오답 이미지 분류하기 위함
                predict_label = sess.run(tf.argmax(predictions,1))
                actual_label = sess.run(tf.argmax(ALL_TEST_LABELS, 1))
                test_result = sess.run(ensemble_correct_prediction)
                false_index = np.squeeze(np.argwhere(test_result == False))
                # print("총 False의 갯수는 {}".format(len(false_index)))
                for i,f_index in enumerate(false_index,1):
                    # print("실제 라벨은: {}, 예측라벨은: {} => {}".format(actual_label[f_index], predict_label[f_index], actual_label[f_index] == predict_label[f_index]))

                    img = test_x_batch[f_index]*255
                    img = np.uint8(img)
                    img = img.reshape(224,224)
                    img = Image.fromarray(img)
                    os.makedirs("test_image_result/false/{}/actual_{}".format(epoch, actual_label[f_index]), exist_ok=True)

                    ImageSave(img,filename="test_image_result/false/{}/actual_{}/{}_actual_{}_predict_{}_softmax_{}"
                              .format(epoch, actual_label[f_index], i,
                                      actual_label[f_index],
                                      predict_label[f_index],
                                      softmax_predictions[f_index]))

            # print('Testing Finished!')
            mon_acuuracy_list[len(mon_acuuracy_list)-1].append(round((1.0-TEST_ACCURACY)*100,3))
            # [[2.4027903079986572, 2.4005317687988281, 2.3938455581665039, 2.3831737041473389]]['model1']
            drawnow(monitorAccuracy, epoch_num=epoch)

            actual_confusionMatrix = onehot2label(ALL_TEST_LABELS)
            prediction_confusionMatrix = onehot2label(predictions)
            if len(temp) == 2 and temp[1] > temp[0] and temp[1] > 0.4:
                confusion_matrix = ConfusionMatrix(actual_confusionMatrix, prediction_confusionMatrix)
                print(confusion_matrix)

        if epoch == 100:
            drawnow(monitorAccuracy, epoch_num=epoch, pltSave=True)
            break
            # confusion_matrix.print_stats()

            # 모델 저장이 필요할 때만 활성화 함
            # TEST_ACCURACY_LIST.append(TEST_ACCURACY)
            # if len(TEST_ACCURACY_LIST) != 1:
            #     if float(TEST_ACCURACY_LIST[0]) >= float(TEST_ACCURACY_LIST[1]) :
            #         print("이전 정확도: {}, 현재 정확도:{} ".format(TEST_ACCURACY_LIST[0], TEST_ACCURACY_LIST[1]))
            #         print("Ealry Stop 으로 학습을 중단합니다.")
            #         print("최고정확도 {}".format(TEST_ACCURACY_LIST[0]))
            #         break
            #     else:
            #         print("이전 정확도: {}, 현재 정확도:{} ".format(TEST_ACCURACY_LIST[0], TEST_ACCURACY_LIST[1]))
            #         TEST_ACCURACY_LIST[0] = TEST_ACCURACY_LIST[1]
            #         TEST_ACCURACY_LIST.pop()
            #         saver.save(sess, 'log/epoch_' + str(LAST_EPOCH) + '.ckpt')
            #         print("학습을 계속 진행합니다.")

    # drawnow(monitorTrainCost, pltSave=True)
    # drawnow(monitorAccuracy, pltSave=True)


    print('Learning Finished!')

    # 종료 시간 체크
    etime = time.time()
    c_time = round(etime - stime, 6)/60
    p_time = predictConsumtionTime(LAST_EPOCH-1)

    print('Total Consumption Time : {} 분'.format(c_time))
    print('학습에 소요된 Consumption Time : 약 {} 분'.format(p_time))
    print('Early Stoping에 소요된 Consumption Time : 약 {} 분'.format(c_time-p_time))






