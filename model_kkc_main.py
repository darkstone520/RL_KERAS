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
        train_last_index = round(TRAIN_RATE*len(lines))
        file.close()
        return lines[:train_last_index], lines[train_last_index:]

def readBatchData(lines, START_BATCH_INDEX):
    # data = np.loadtxt(data, dtype=np.uint8)
    data = [line.split(',')[:-1] for line in lines]
    data = np.array(data, dtype=np.float32)
    data, label = data[START_BATCH_INDEX:START_BATCH_INDEX+BATCH_SIZE,:-1], data[START_BATCH_INDEX:START_BATCH_INDEX+BATCH_SIZE:,-1]
    data = data/255.
    label = [ [1,0] if label == 0 else [0,1] for label in label.tolist()]
    label = np.array(label)
    START_BATCH_INDEX += BATCH_SIZE
    return data, label



__DATA_PATH = "preprocessed_data/"

IMG_SIZE = (144,144)
BATCH_SIZE = 100
START_BATCH_INDEX = 0
TRAIN_EPOCHS = 20
TRAIN_RATE = 0.8
NUM_MODELS = 5
LEARNING_RATE = 0.0001

TRAIN_DATA, TEST_DATA = loadInputData()


print("Session open")
# initialize
sess = tf.Session()

models = []
num_models = 2
for m in range(num_models):
    models.append(Model(sess, "model" + str(m)))

sess.run(tf.global_variables_initializer())

print('Learning Started!')

# train my model
for epoch in range(1):#range(TRAIN_EPOCHS):
    avg_cost_list = np.zeros(len(models))
    total_batch = math.trunc(int(len(TRAIN_DATA) / BATCH_SIZE))

    for i in range(total_batch):
        BATCH_DATA = TRAIN_DATA[START_BATCH_INDEX:START_BATCH_INDEX+BATCH_SIZE]
        print("Batch Data Reading {}".format(i))
        TRAIN_DATA_X, TRAIN_DATA_Y = readBatchData(BATCH_DATA,START_BATCH_INDEX)

        # train each model
        for m_idx, m in enumerate(models):
            c, _ = m.train(TRAIN_DATA_X, TRAIN_DATA_Y)
            avg_cost_list[m_idx] += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost_list)

print('Learning Finished!')
START_BATCH_INDEX = 0
TEST_EPHOCS=1
print('Testing Started!')

ensemble_accuracy = 0.
model_accuracy = [0., 0.]
cnt = 0
TOTAL_BATCH = math.trunc(len(TEST_DATA)/BATCH_SIZE)
for _ in range(TEST_EPHOCS):
    for i in range(TOTAL_BATCH):
        print("Batch Data Reading {}/{}".format(i, TOTAL_BATCH))
        test_x_batch, test_y_batch = readBatchData(TEST_DATA, START_BATCH_INDEX)
        test_size = len(test_y_batch)
        predictions = np.zeros(test_size * 2).reshape(test_size, 2)

        model_result = np.zeros(test_size * 2, dtype=np.int).reshape(test_size, 2)
        model_result[:, 0] = range(0, test_size)

        for idx, m in enumerate(models):
            model_accuracy[idx] += m.get_accuracy(test_x_batch, test_y_batch)
            p = m.predict(test_x_batch)
            model_result[:, 1] = np.argmax(p, 1)
            for result in model_result:
                predictions[result[0], result[1]] += 1

        ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(test_y_batch, 1))
        ensemble_accuracy += tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
        cnt += 1

for i in range(len(model_accuracy)):
    print('Model ' + str(i) + ' : ', model_accuracy[i] / cnt)
print('Ensemble Accuracy : ', sess.run(ensemble_accuracy) / cnt)
print('Testing Finished!')

