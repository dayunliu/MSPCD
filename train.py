#  -*-encoding: utf-8 -*-
import pandas as pd
import numpy as np
import csv
from base_model import get_model
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from keras.optimizers import Adam
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score

np.random.seed(1)


def get_circ_list():
    circ_path = "../data/circFunbase/circFunbase_circRNA.csv"
    circ_list = []
    csv_reader = csv.reader(open(circ_path))
    for circ_name in csv_reader:
        circ_list.extend(circ_name)
    return circ_list


def get_dis_list():
    disease_path = "../data/circFunbase/circFunbase_disease.csv"
    disease_list = []
    csv_reader = csv.reader(open(disease_path))
    for disease_name in csv_reader:
        disease_list.extend(disease_name)
    return disease_list


def get_all_samples():
    path = "../data/circFunbase/circRNA_disease.csv"
    circ_dis = pd.read_csv(path)
    circ_dis = circ_dis.set_index("Unnamed: 0")
    circ_list = get_circ_list()
    dis_list = get_dis_list()
    positives_list = []
    negatives_list = []
    for circ in circ_list:
        for dis in dis_list:
            if circ_dis.loc[circ][dis] == 0:
                sample = [circ, dis]
                negatives_list.append(sample)
            else:
                sample = [circ, dis]
                positives_list.append(sample)
    return positives_list, negatives_list


def get_train_samples():
    positive_list, negative_list = get_all_samples()
    pos_lg = len(positive_list)
    Y = [1 for _ in range(0, pos_lg)]
    Y.extend([0 for _ in range(0, pos_lg)])
    X = positive_list
    for i in range(pos_lg):
        X.append(negative_list[i * 20])
    return X, Y


def index_to_ver(index_list):
    circ_sim_path = "../data/circFunbase/circRNA_sim.csv"
    circ_sim = pd.read_csv(circ_sim_path)
    circ_sim = circ_sim.set_index("Unnamed: 0")
    dis_sim_path = "../data/circFunbase/disease_sim.csv"
    dis_sim = pd.read_csv(dis_sim_path)
    dis_sim = dis_sim.set_index("Unnamed: 0")

    circ_ver = []
    dis_ver = []
    for index in index_list:
        circ_name = index[0]
        circ_ver.append(circ_sim.loc[circ_name].values[:])
        dis_name = index[1]
        dis_ver.append(dis_sim.loc[dis_name].values[:])
    return circ_ver, dis_ver


def train():
    X, Y = get_train_samples()
    X = np.array(X)
    Y = np.array(Y)
    X, Y = shuffle(X, Y, random_state=1)
    kf = KFold(n_splits=5)
    AUC_list = []
    acc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    for train_index, test_index in kf.split(X, Y):
        X_train = X[train_index]
        Y_train = Y[train_index]
        X_test = X[test_index]
        Y_test = Y[test_index]
        circrna_vers, disease_vers = index_to_ver(X_train)
        circrna_test_vers, dis_test_vers = index_to_ver(X_test)
        model = get_model()
        opt = Adam(lr=0.001, decay=1e-3 / 200)
        model.compile(loss="binary_crossentropy", optimizer=opt)
        model.fit([circrna_vers, disease_vers], Y_train, verbose=0, epochs=100, batch_size=256, validation_split=0.1,
                  shuffle=False)
        predict_value = model.predict([circrna_test_vers, dis_test_vers])
        AUC = metrics.roc_auc_score(Y_test, predict_value)
        acc = metrics.accuracy_score(Y_test, predict_value.round())
        p = precision_score(Y_test, predict_value.round())
        r = recall_score(Y_test, predict_value.round())
        f1 = f1_score(Y_test, predict_value.round())
        print("the AUC is ", AUC)
        print("the acc is ", acc)
        print("the precision is ", p)
        print("the recall is ", r)
        print("the f1_score is ", f1)
        AUC_list.append(AUC)
        precision_list.append(p)
        recall_list.append(r)
        acc_list.append(acc)
        f1_list.append(f1)
    print("the average of the AUC is ", sum(AUC_list) / len(AUC_list))
    print("the average of the acc is ", sum(acc_list) / len(acc_list))
    print("the average of the precision is ", sum(precision_list) / len(precision_list))
    print("the average of the recall is ", sum(recall_list) / len(recall_list))
    print("the average of the f1_score is ", sum(f1_list) / len(f1_list))


if __name__ == "__main__":
    train()
