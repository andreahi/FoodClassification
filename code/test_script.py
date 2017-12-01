import os
from time import sleep

import keras.backend as K
import numpy as np  # linear algebra
import pandas as pd
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.engine import Input, Model
from keras.layers import AveragePooling2D
from keras.layers import Dropout, Flatten
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from sklearn.metrics import fbeta_score, confusion_matrix

import config
import util
from code.Augmentor import Augmentor

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152


def get_model():
    return get_pretrained()


def findClassBalanceWeights(p_train, y_train):
    print "p_train: ", p_train
    print "y_train: ", y_train

    scores = []
    thresholds = []
    for i in range(0, 5000):
        t = i / 5000.0
        p_train_t = threshold_data(p_train, threshold=t)
        score = fbeta_score(y_train, p_train_t, beta=1)
        scores.append(score)
        thresholds.append(t)
    print "scores: ", scores
    print "max : ", np.max(scores, axis=0)
    print "mean max : ", np.mean(np.max(scores, axis=0))

    return thresholds[np.argmax(scores)]


def threshold_data(p_train, threshold=0.5):
    ret = np.copy(p_train)
    ret[ret > threshold] = 1
    ret[ret <= threshold] = 0

    return ret



def get_pretrained():
    from keras.applications.inception_v3 import InceptionV3
    from keras.layers import Dense

    base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(299, 299, 3)))
    x = base_model.output
    x = AveragePooling2D(pool_size=(8, 8))(x)
    x = Dropout(.01)(x)
    x = Flatten()(x)
    predictions = Dense(368, init='glorot_uniform', W_regularizer=l2(.0005), activation='sigmoid')(x)

    model = Model(input=base_model.input, output=predictions)

    model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def get_pretrained_locked(model):

    opt = SGD(lr=.01, momentum=.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def top_1_categorical_accuracy(y_true, y_pred, k=1):
    return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k), axis=-1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def dice_coef(y_true, y_pred):
    smooth = 1.

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def generate_data(x, y):
    while 1:
        indices = np.random.choice(x.shape[0], 50, replace=True)
        yield (x[indices], y[indices])

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def dice_coef(y_true, y_pred):
    smooth = 1.

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def train(name, model, nc, target_class):
    x_train, x_valid, y_train, y_valid  = util.get_data(target_class)
    print x_train.shape
    model_checkpoint = ModelCheckpoint(str(target_class)+"_best.h2", monitor='val_loss', save_best_only=True)
    model_checkpoint_latest = ModelCheckpoint(name + '_latest.h5')
    model_checkpoint_epoch = ModelCheckpoint('models/{epoch:02d}-{acc:.3f}.hdf5')

    try:
        datagen = getDataGenTrain()
        data_gen_validation = getDataGenValidation(x_train)
        if False:
            model.load_weights("test_latest.h5")

        else:
            #model.load_weights("test_latest.h5")


            def schedule(epoch):
                if epoch < 15:
                    return .01
                elif epoch < 28:
                    return .002
                else:
                    return .0004

            lr_scheduler = LearningRateScheduler(schedule)

            history = model.fit_generator(#generate_data(x_train, y_train),
                                      datagen.flow(x_train, np.array(y_train, dtype="float16") , batch_size=16, shuffle=True),
                                      #samples_per_epoch=x_train.shape[0],
                                      steps_per_epoch=4000,
                                      epochs=32,
                                      initial_epoch=0,
                                      #validation_data=(x_valid, y_valid),
                                      #validation_data=(data_gen_validation.flow(x_valid, y_valid)),
                                      #validation_steps=x_valid.shape[0] // 32,
                                      workers=2,
                                      max_q_size=100,
                                      callbacks=[model_checkpoint, model_checkpoint_latest, model_checkpoint_epoch],
                                      verbose=1)


        model = get_pretrained_locked(model)

        def schedule(epoch):
            if epoch < 10:
                return .00008
            elif epoch < 20:
                return .000016
            else:
                return .0000032

        lr_scheduler = LearningRateScheduler(schedule)

        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32, shuffle=True),
                                      steps_per_epoch=4000,
                                      epochs=100,
                                      initial_epoch=0,
                                      workers=2,
                                      max_q_size=100,
                                      callbacks=[lr_scheduler, model_checkpoint, model_checkpoint_latest, model_checkpoint_epoch],
                                      verbose=1)

        print "training done"
    except Exception as exp:
        print "got exception"
        print(exp)
        K.clear_session()
        return -1

    print(history.history)
    model.load_weights("test_latest.h5")

    p_valid = model.predict(x_valid, batch_size=10)


    threshold = 0.5

    print "using threshold ", threshold

    p_valid = threshold_data(p_valid, threshold)

    print_stats(p_valid, y_valid)

def print_stats(p_valid, y_valid):

    zero_accuracy = np.array(sum(y_valid == 0), dtype=float) / len(y_valid)
    print "constant func accuracy", np.array2string(np.max([1 - zero_accuracy, zero_accuracy], axis=0),
                                                    max_line_width=200)
    print "class accuracy: ", np.array2string(np.array(sum(y_valid == p_valid), dtype=float) / len(y_valid),
                                              max_line_width=200)
    print "avg constant func accuracy", np.array2string(np.average(np.max([1 - zero_accuracy, zero_accuracy], axis=0)),
                                                        max_line_width=200)
    print "avg class accuracy: ", np.array2string(
        np.average(np.array(sum(y_valid == p_valid), dtype=float) / len(y_valid)), max_line_width=200)
    print "nr correct: ", np.array2string(sum(y_valid == p_valid), max_line_width=200)
    print "nr wrong: ", np.array2string(sum(y_valid != p_valid), max_line_width=200)
    print
    print "true possitive:", np.array2string(sum(np.logical_and(y_valid == p_valid, y_valid == 1)), max_line_width=200)
    print "true negative: ", np.array2string(sum(np.logical_and(y_valid == p_valid, y_valid == 0)), max_line_width=200)
    print "false positie: ", np.array2string(sum(np.logical_and(y_valid != p_valid, y_valid == 0)), max_line_width=200)
    print "false negative:", np.array2string(sum(np.logical_and(y_valid != p_valid, y_valid == 1)), max_line_width=200)

def pre_process(img):
    return img/255.0


def getDataGenTrain():
    datagen = Augmentor(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
        fill_mode="reflect",
        zoom_range=[0.5,1.0],
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,
        preprocessing_function=pre_process,
        channel_shift_range=30)  # randomly flip images


    return datagen

def getDataGenValidation(x_train):
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0.1,  # randomly rotate images in the range (degrees, 0 to 180)
        fill_mode="reflect",
        width_shift_range=0.0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(x_train[0:2000])

    return datagen


def run(net_config, target_class, fresh=True):
    model = get_model()
    train("test", model, net_config, target_class)


def test(net_config, target_class):
    model = get_model()
    x_train, x_valid, y_train, y_valid  = util.get_data(target_class)


    model.load_weights("test_latest.h5")

    predict = model.predict(x_valid)

    p_valid = predict
    max_index_p_valid = np.argmax(p_valid, axis=1)
    max_index_y_valid = np.argmax(y_valid, axis=1)

    print "acc score: ", np.sum(max_index_p_valid == max_index_y_valid) / float(len(max_index_p_valid))

    for idx, e in enumerate(max_index_p_valid):
        p_valid[idx,:] = 0
        p_valid[idx,e] = 1

    print "max _index ", max_index_p_valid
    print "predict: ", predict
    print "y_valid: ", y_valid

    print_stats(p_valid, y_valid)

def validation(net_config, target_class):

    import os


    model = get_model()
    x_train, x_valid, y_train, y_valid  = util.get_data(target_class)

    np_sum = np.sum(y_valid, axis=0)
    print "len axis sum: ", len(np_sum)
    print "axis sum: ", np_sum

    import pickle
    favorite_color = pickle.load(open("label_mapping.p", "rb"))
    print favorite_color
    inv_map = {v: k for k, v in favorite_color.items()}

    crossmap_list = pd.read_csv('../input/crossmapping.csv',  header=None).values
    crossmap = {}
    for e in crossmap_list:
        crossmap[e[1]] = e[0]
    print crossmap

    matrix = np.loadtxt('file.txt')
    print "max confusion: ", np.unravel_index(matrix.argmax(), matrix.shape)
    confusions_printed = 0
    for _ in range(1500):
        index = np.unravel_index(matrix.argmax(), matrix.shape)
        max_val = matrix[index[0]][index[1]]
        matrix[index[0]][index[1]] = 0

        if index[0] == index[1]:
            continue
        name1 = inv_map[index[0]]
        name2 = inv_map[index[1]]
        if name1 in crossmap:
            continue
        print index, max_val
        print name1, name2, max_val
        confusions_printed += 1
        if confusions_printed > 20:
            break

    x_valid_preprocessed = pre_process(x_valid)

    while True:

        listdir = os.listdir("models")
        if not listdir:
            sleep(1000)
            continue
        oldest = min(["models/" + e for e in listdir], key=os.path.getctime)
        print oldest

        model.load_weights(oldest)

        predict = model.predict(x_valid_preprocessed)


        p_valid = predict #np.mean(np.concatenate([predict], axis=1), axis=1)
        max_index_p_valid = np.argmax(p_valid, axis=1)
        max_index_y_valid = np.argmax(y_valid, axis=1)
        matrix = confusion_matrix(max_index_p_valid, max_index_y_valid)
        #print matrix
        np.savetxt('file.txt', matrix)
        valid_acc = np.sum(max_index_p_valid == max_index_y_valid) / float(len(max_index_p_valid))
        print "acc score: ", valid_acc
        os.rename(oldest, "donemodels/" + str(valid_acc) + ".h5")

        p_valid = threshold_data(predict, 0.5)

        print "avg class accuracy: ", np.array2string(
            np.average(np.array(sum(y_valid == p_valid), dtype=float) / len(y_valid)), max_line_width=200)


def exp(net_config, target_class):
    x_train, x_valid, y_train, y_valid  = util.get_data(target_class)
    #print x_train
    datagen = getDataGenTrain()

    model = get_model()
    model.load_weights("donemodels/"+"0.552645073071.h5")
    model.save('my_model.h5')

    import pickle
    favorite_color = pickle.load(open("label_mapping.p", "rb"))
    print favorite_color
    inv_map = {v: k for k, v in favorite_color.items()}

    model_predict = model.predict(pre_process(x_valid))
    wrong_matrix  = {}
    total_matrix  = {}
    for i in range(len(model_predict)):
        predict_thresholded = model_predict[i] > 0.5
        y_valid_threshold = y_valid[i] > 0.5
        for j in range(len(predict_thresholded)):
            if predict_thresholded[j] and not y_valid_threshold[j]:
                indices = [i for i, x in enumerate(y_valid_threshold) if x]
                for e in indices:
                    j_name = inv_map[j]
                    e_name = inv_map[e]
                    if not predict_thresholded[e]:

                        if not (j_name, e_name) in wrong_matrix:
                            wrong_matrix[(j_name, e_name)] = 0
                        wrong_matrix[(j_name, e_name)] += 1

                    if not (j_name, e_name) in total_matrix:
                        total_matrix[(j_name, e_name)] = 0
                    total_matrix[(j_name, e_name)] += 1

    print "wrong matrix :"
    print wrong_matrix
    print "correct matrix:"
    print total_matrix

    for key in wrong_matrix:
        print key, wrong_matrix[key]/float(total_matrix[key])

    return
    for e in model_predict:
        print e

    for x, y in zip(x_valid, y_valid):
        flow = datagen.flow(np.array([x] * 10), np.array([y] * 10), batch_size=10)
        x_augmented, y_augmented = flow.next()
        for (x1, y1) in zip(x_augmented, y_augmented):
            showImage((x1).astype('uint8'), str(y1))

    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
        print x_batch[0]
        showImage((x_batch[0]).astype('uint8'), str(y_batch[0]))

    inputs = []
    targets = []
    for i in range(1000):
        inputs.append(np.fliplr(np.copy(x_valid[i])))
        targets.append(y_valid[i])
    x_valid1 = np.array(inputs)
    y_valid = np.array(targets)

    inputs = []
    for i in range(1000):
        inputs.append((np.copy(x_valid[i])))
    x_valid2 = np.array(inputs)

    predict = np.add(model.predict(x_valid1), model.predict(x_valid2))

    print predict

    p_valid = predict
    max_index_p_valid = np.argmax(p_valid, axis=1)
    max_index_y_valid = np.argmax(y_valid, axis=1)

    print "acc score: ", np.sum(max_index_p_valid == max_index_y_valid) / float(len(max_index_p_valid))



if __name__ == "__main__":
    import sys
    print(sys.argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    if sys.argv[2] == "train":
        run(config.net_config, 4, fresh=True)
    if sys.argv[2] == "test":
        test(config.net_config, 4)
    if sys.argv[2] == "validation":
        validation(config.net_config, 4)
    if sys.argv[2] == "exp":
        exp(config.net_config, 4)

