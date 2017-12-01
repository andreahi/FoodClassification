import numpy
import os
import pickle

import cv2
import numpy
import numpy as np
import pandas as pd
from skimage import transform
from tqdm import tqdm

INPUT_SHAPE = (299, 299, 3)

def get_data(target_class, split=35000):


    crossmap_list = pd.read_csv('../input/crossmapping.csv',  header=None).values
    crossmap = {}
    for e in crossmap_list:
        crossmap[e[1]] = e[0]
    print crossmap

    filename_x = '../input/'+str(INPUT_SHAPE[0])+str(INPUT_SHAPE[1])+'jpg_data_x.h5'
    filename_y = '../input/'+str(target_class)+'_jpg_data_y.h5'


    label_mapping = {}

    if  not os.path.exists(filename_x):

        train_data_x = np.memmap(filename_x, dtype='float16', mode='w+', shape=(9866+75749+31395, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]))
        train_data_y = np.memmap(filename_y, dtype='ubyte', mode='w+', shape=(9866+75749+31395, 368))

        i = 0
        i = get_food256(i, label_mapping, train_data_x, train_data_y)
        i = get_food11(i, label_mapping, train_data_x, train_data_y)
        get_food101(i, label_mapping, train_data_x, train_data_y)

        pickle.dump(label_mapping, open("label_mapping.p", "wb"))

        shuffle_in_unison( train_data_x, train_data_y)
        train_data_x.flush()
        train_data_y.flush()


    label_mapping = pickle.load(open("label_mapping.p", "rb"))
    inv_map = {v: k for k, v in label_mapping.items()}
    fix_crossmapp(crossmap, label_mapping, filename_y, inv_map)


    x = np.memmap(filename_x, dtype='float16', mode='r', shape=(9866+75749+31395, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]))
    y = np.memmap(filename_y, dtype='ubyte', mode='r', shape=(9866+75749+31395, 368))



    split = int(len(x) * 0.8)
    x_train = x[:split]
    x_valid = x[split:]
    y_train = y[:split]
    y_valid = y[split:]

    print y_train
    count_train = np.sum(y_train, axis=0)
    count_valid = np.sum(y_valid, axis=0)

    print "count_train  ", count_train
    print "count_valid  ", count_valid

    print "size train ", len(y_train)
    print "size valid ", len(y_valid)

    print "shape ", y_valid.shape
    #print y_train

    return x_train, x_valid, y_train, y_valid


def fix_crossmapp(crossmap, favorite_color, filename_y, inv_map):
    y_tmp = np.memmap(filename_y, dtype='ubyte', mode='readwrite', shape=(9866 + 75749 + 31395, 368))
    argmax = np.argmax(y_tmp, axis=1)
    print "argmarx: ", argmax
    print "len argmax: ", len(argmax)
    print "inv map", inv_map
    print 293 in argmax
    for i in range(len(argmax)):
        e = argmax[i]
        label_text = inv_map[e]
        if label_text in 'croque_madame':
            print "label_text: ", label_text
        if label_text in crossmap:
            print label_text, crossmap[label_text]
            first = favorite_color[label_text]
            second = favorite_color[crossmap[label_text]]
            print first, second
            y_tmp[i][first] = 1
            y_tmp[i][second] = 1
    y_tmp.flush()


def get_food256(i, label_mapping, train_data_x, train_data_y):
    rootdir = '../input/UECFOOD256'
    df_train = pd.read_csv('../input/UECFOOD256/category.txt')
    labels = {}
    for e in df_train.values:
        e__split = e[0].split()
        labels[e__split[0]] = " ".join(e__split[1:])
    for subdir, dirs, files in os.walk(rootdir):
        for f in tqdm(files, miniters=1000):
            filepath = os.path.join(subdir, f)
            if ".txt" in filepath:
                continue
            label = filepath.split('/')[3]


            label = labels[label]

            if label not in label_mapping:
                label_mapping[label] = len(label_mapping)

            img_path = filepath
            img = cv2.imread(img_path)
            train_data_x[i, :, :, :] = np.array(preprocess_img(img), np.float16)

            targets = np.zeros(368, np.ubyte)
            targets[label_mapping[label]] = 1
            train_data_y[i, :] = targets
            i += 1
    return i


def get_food101(i, label_mapping, train_data_x, train_data_y):

    df_train = pd.read_csv('../input/images/train.txt')
    for f in tqdm(df_train.values, miniters=1000):
        label = f[0].split('/')[0]


        if label not in label_mapping:
            label_mapping[label] = len(label_mapping)



        img_path = '../input/images/' + f[0] + '.jpg'
        img = cv2.imread(img_path)
        train_data_x[i, :, :, :] = np.array(preprocess_img(img), np.float16)

        targets = np.zeros(368, np.ubyte)
        targets[label_mapping[label]] = 1
        train_data_y[i, :] = targets
        i += 1


def get_food11(i, label_mapping, train_data_x, train_data_y):
    labels = {0: "Bread", 1: "Dairy product", 2: "Dessert", 3: "Egg", 4: "Fried food", 5: "Meat",
              6: "Noodles/Pasta", 7: "Rice", 8: "Seafood", 9: "Soup", 10: "Vegetable/Fruit"}

    listdir = os.listdir("../input/Food11/training/")
    for f in tqdm(listdir, miniters=1000):
        label = int(f.split('_')[0])
        label = labels[label]
        # print "f: ", f
        # print "label: ", label
        if label not in label_mapping:
            label_mapping[label] = len(label_mapping)



        img_path = '../input/Food11/training/' + f
        img = cv2.imread(img_path)
        train_data_x[i, :, :, :] = np.array(preprocess_img(img), np.float16)

        targets = np.zeros(368, np.ubyte)
        targets[label_mapping[label]] = 1
        train_data_y[i, :] = targets
        i += 1
    return i

def shuffle_in_unison(a, b):
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(a)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(b)

def preprocess_img(img):
    # Histogram normalization in v channel
    #hsv = color.rgb2hsv(img)
    #hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    #img = color.hsv2rgb(hsv)

    # central square crop
    #min_side = min(img.shape[:-1])
    #centre = img.shape[0] // 2, img.shape[1] // 2
    #img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
    #          centre[1] - min_side // 2:centre[1] + min_side // 2,
    #          :]

    # rescale to standard size
    img = transform.resize(img, (INPUT_SHAPE[0], INPUT_SHAPE[1]), preserve_range=True)

    return img




if __name__ == "__main__":
    get_data()