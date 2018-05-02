# KNN
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import noise
import proba_noise
from sklearn.model_selection import StratifiedKFold

def KClassifier(method,norm,selection):
    print('---------------------------------------------------------------------------------------------------')
    print('KNN')
    acc_save = []
    f1_save = []
    auc_save = []
    dataset = np.loadtxt('new.csv', delimiter=',')
    [a, b] = dataset.shape
    dataset_label = dataset[:, b - 1]
    dataset_train = dataset[:, 0:b - 1]
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for train_index, dev_index in skf.split(dataset_train, dataset_label):
        X_cv_train, X_cv_dev = dataset_train[train_index], dataset_train[dev_index]
        y_cv_train, y_cv_dev = dataset_label[train_index], dataset_label[dev_index]
        if norm == 3:
            proba_noise.pre_select_data(selection, 1)
        else:
            noise.pre_select_data(dataset, 1, selection)
        train = np.loadtxt('B_Trainset_data.csv', delimiter=',')
        label_train = np.loadtxt('B_Trainset_label.csv', delimiter=',')
        test = np.loadtxt('Test_data.csv', delimiter=',')
        label_test = np.loadtxt('Test_label.csv', delimiter=',')
        KNN = KNeighborsClassifier(n_neighbors=100)
        KNN.fit(train, label_train)
        y_pred_train = KNN.predict(train)
        acc_train = accuracy_score(y_pred_train, label_train)
        print('acc_train = ',acc_train)
        target = ['class 1', 'class 2']
        print(classification_report(label_train, y_pred_train, target_names=target))
        print('auc = ', roc_auc_score(label_train, y_pred_train, average='weighted'))
        y_pred_test = KNN.predict(test)
        # y_prob_pred_test = suppvec.predict_proba(test)
        acc_test = accuracy_score(y_pred_test, label_test)
        print('acc_test = ',acc_test)
        acc_save.append(acc_test)
        target = ['class 1', 'class 2']
        print(classification_report(label_test, y_pred_test, target_names=target))
        f1 = f1_score(label_test, y_pred_test, average='weighted')
        print('f1 score =', f1)
        f1_save.append(f1)
        auc = roc_auc_score(label_test, y_pred_test, average='weighted')
        print('auc = ', auc)
        auc_save.append(auc)
        fpr, tpr, thresholds = roc_curve(label_test, y_pred_test)
        plt.plot(fpr, tpr)
        plt.xlabel('fpr')
        plt.ylabel('tpr')
        plt.show()
        print('----------------------------------------------------------------------------------------------------')
    print('5 iteration average results')
    print('test_acc = ', np.mean(acc_save))
    print('f1_score = ', np.mean(f1_save))
    print('auc = ', np.mean(auc_save))
