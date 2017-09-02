from model_kkc import Model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math


def plot_image(image):
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
    print("Loading Data")
    with open(__DATA_PATH + "cat_dog_data", "r", encoding="utf-8") as file:
        # lines : 모든 lines(데이터행)을 불러온다.
        lines = file.readlines()
        lines = random.sample(lines, len(lines))
        train_last_index = round(TRAIN_RATE * len(lines))
        file.close()
        return lines[:train_last_index], lines[train_last_index:]


def readBatchData(lines):
    data = [line.split(',')[:-1] for line in lines]
    data = np.array(data, dtype=np.float32)
    data, label = data[:, :-1], data[:, -1]
    data = data / 255.
    label = [[1, 0] if label == 0 else [0, 1] for label in label.tolist()]
    label = np.array(label)
    return data, label

def shuffleBatchLines(lines):
    lines = random.sample(lines, BATCH_SIZE)
    return lines


__DATA_PATH = "preprocessed_data/"

IMG_SIZE = (144, 144)
BATCH_SIZE = 100
TRAIN_EPOCHS = 20
TRAIN_RATE = 0.8
NUM_MODELS = 3
LEARNING_RATE = 0.005

TRAIN_DATA, TEST_DATA = loadInputData()

print("Session open")
# initialize
sess = tf.Session()

models = []
for m in range(NUM_MODELS):
    models.append(Model(sess, "model" + str(m)))

sess.run(tf.global_variables_initializer())

print('Learning Started!')

# train my model
for epoch in range(TRAIN_EPOCHS):

    avg_cost_list = np.zeros(len(models))
    total_batch_num = math.trunc(int(len(TRAIN_DATA) / BATCH_SIZE))

    for i in range(total_batch_num):

        print("Batch Data Reading {}/{}".format(i + 1, total_batch_num))
        batch_data = shuffleBatchLines(TRAIN_DATA)
        train_x_batch, train_y_batch = readBatchData(batch_data)

        # train each model
        for m_idx, m in enumerate(models):
            c, _ = m.train(train_x_batch, train_y_batch)
            avg_cost_list[m_idx] += c / total_batch_num

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost_list)
    START_BATCH_INDEX = 0

print('Learning Finished!')

print('Testing Started!')
TEST_EPHOCS = 1
ENSEMBLE_ACCURACY = 0.
MODEL_ACCURACY = [0., 0., 0.]
CNT = 0

for _ in range(TEST_EPHOCS):

    total_batch_num = math.trunc(len(TEST_DATA) / BATCH_SIZE)

    for i in range(total_batch_num):

        print("Batch Data Reading {}/{}".format(i + 1, total_batch_num))
        batch_data = shuffleBatchLines(TEST_DATA)
        test_x_batch, test_y_batch = readBatchData(batch_data)

        test_size = len(test_y_batch)
        predictions = np.zeros(test_size * 2).reshape(test_size, 2)
        model_result = np.zeros(test_size * 2, dtype=np.int).reshape(test_size, 2)
        model_result[:, 0] = range(0, test_size)

        for idx, m in enumerate(models):
            MODEL_ACCURACY[idx] += m.get_accuracy(test_x_batch, test_y_batch)
            p = m.predict(test_x_batch)
            model_result[:, 1] = np.argmax(p, 1)
            for result in model_result:
                predictions[result[0], result[1]] += 1

        ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(test_y_batch, 1))
        ENSEMBLE_ACCURACY += tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
        CNT += 1

for i in range(len(MODEL_ACCURACY)):
    print('Model ' + str(i) + ' : ', MODEL_ACCURACY[i] / CNT)
print('Ensemble Accuracy : ', sess.run(ENSEMBLE_ACCURACY) / CNT)
print('Testing Finished!')

