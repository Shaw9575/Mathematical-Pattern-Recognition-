# Gaussian Bayes classify
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import noise
import weight
import proba_noise
import proba_weight


def GBC(method, norm, selection):
    print('---------------------------------------------------------------------------------------------------')
    print('Bayes')
    acc_save = []
    f1_save = []
    auc_save = []
    dataset = np.loadtxt('new.csv', delimiter=',')
    [a, b] = dataset.shape
    dataset_label = dataset[:, b-1]
    dataset_train = dataset[:, 0:b - 1]
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for train_index, dev_index in skf.split(dataset_train, dataset_label):
        X_cv_train, X_cv_dev = dataset_train[train_index], dataset_train[dev_index]
        y_cv_train, y_cv_dev = dataset_label[train_index], dataset_label[dev_index]
        if norm == 3:
            if method == 1:
                proba_noise.pre_select_data(selection,1)
                ratio = 1
            if method == 2:
                [ratio, sw] = proba_weight.pre_select_data(selection,1)
        else:
            if method == 1:
                noise.pre_select_data(dataset, 1,selection)
                ratio = 1
            if method == 2:
                [ratio, sw] = weight.pre_select_data(dataset, 1,selection)
        train = np.loadtxt('Bayes.csv', delimiter=',')
        label_train = np.loadtxt('Bayes_label.csv', delimiter=',')
        test = np.loadtxt('BayesTest.csv', delimiter=',')
        label_test = np.loadtxt('Bayes_TL.csv', delimiter=',')
        gnb = GaussianNB()
        gnb.fit(train, label_train)
        y_pred_train = gnb.predict(train)
        acc_train = accuracy_score(y_pred_train, label_train)
        print('acc_train = ',acc_train)
        # f1
        target = ['class 1', 'class 2']
        print(classification_report(label_train, y_pred_train, target_names=target))
        # auc
        # fpr, tpr, thresholds = metrics.roc_curve(label_train, y_pred_train, pos_label=2)
        print('auc = ', roc_auc_score(label_train, y_pred_train, average='weighted'))
        y_pred_test = gnb.predict(test)
        y_prob_pred_test = gnb.predict_proba(test)
        acc_test = accuracy_score(y_pred_test, label_test)
        print('acc_test = ',acc_test)
        acc_save.append(acc_test)
        target = ['class 1', 'class 2']
        print(classification_report(label_test, y_pred_test, target_names=target))
        f1 = f1_score(label_test, y_pred_test, average='weighted')
        print('f1 score =', f1)
        f1_save.append(f1)
        auc = roc_auc_score(label_test, y_pred_test, average='weighted')
        print('auc =', auc)
        auc_save.append(auc)
        fpr, tpr, thresholds = roc_curve(label_test, y_prob_pred_test[:, 1])
        plt.plot(fpr, tpr)
        plt.xlabel('fpr')
        plt.ylabel('tpr')
        plt.show()
        print('---------------------------------------------------------------------------------------------------')
    print('5 iterations average results')
    print('test_acc = ', np.mean(acc_save))
    print('f1_score = ', np.mean(f1_save))
    print('auc = ', np.mean(auc_save))