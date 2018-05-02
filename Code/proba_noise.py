# Pre-processing
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import shuffle
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import f_classif

def pre_select_data(selection,norm):
    Trainset = np.loadtxt('Trainset.csv', delimiter=',')
    (train_num, b) = Trainset.shape
    feature = b - 1
    Test = np.loadtxt('Test.csv', delimiter=',')
    test_num = Test.shape[0]
    Train_label = Trainset[:, feature]
    Train_info = Trainset[:, 0:feature]
    Test_info = Test[:,0:feature]

    if selection == 1:
        fs = mutual_info_classif(X=Train_info, y=Train_label)
        count3 = 0
        for i in range(0, feature):
            if fs[i] == 0:
                print(i)
                count3 = count3 + 1
        data_new = np.zeros((train_num, b - count3))
        test_new = np.zeros((test_num, b- count3))
        count4 = 0
        for i in range(0, feature):
            if fs[i] != 0:
                data_new[:, count4] = Trainset[:, i]
                test_new[:, count4] = Test[:,i]
                count4 = count4 + 1
        feature = count4
        data_new[:, feature] = Train_label
        test_new[:, feature] = Test[:, b-1]
        print('feature = ', feature)
    if selection == 2:
        clf = ExtraTreesClassifier()
        clf = clf.fit(Train_info, Train_label)
        model = SelectFromModel(clf, prefit=True)
        Train_info = model.transform(Train_info)
        Test_info = model.transform(Test_info)
        feature = Train_info.shape[1]
        data_new = np.zeros((train_num, feature+1))
        test_new = np.zeros((test_num, feature+1))
        data_new[:,0:feature] = Train_info
        data_new[:, feature] = Train_label
        test_new[:, 0:feature] = Test_info
        test_new[:, feature] = Test[:,b-1]
        print('feature = ', feature)
    if selection == 3:
        (us,fs) = f_classif(X=Train_info, y=Train_label)
        count3 = 0
        for i in range(0, feature):
            if fs[i] >= 0.05:
                print(i)
                count3 = count3 + 1
        data_new = np.zeros((train_num, b - count3))
        test_new = np.zeros((test_num, b - count3))
        count4 = 0
        for i in range(0, feature):
            if fs[i] < 0.05:
                data_new[:, count4] = Trainset[:, i]
                test_new[:, count4] = Test[:, i]
                count4 = count4 + 1
        feature = count4
        data_new[:, feature] = Train_label
        test_new[:, feature] = Test[:, b - 1]
        np.savetxt('dd.csv', data_new, delimiter=',')
        print('feature = ', feature)
    if selection == 0:
        feature = b - 1
        data_new = Trainset
        test_new = Test

    Train_data = data_new[:, 0:feature]
    Train_label = data_new[:, feature]
    Test_data = test_new[:, 0:feature]
    Test_label = test_new[:, feature]

    np.savetxt('Bayes_label.csv', Train_label, delimiter=',')
    np.savetxt('Bayes.csv', Train_data, delimiter=',')
    np.savetxt('BayesTest.csv', Test_data, delimiter=',')
    np.savetxt('Bayes_TL.csv', Test_label, delimiter=',')

    if norm == 1:
        scaler = StandardScaler()
        scaler.fit(Train_data)
        Train_data = scaler.transform(Train_data)
        Test_data = scaler.transform(Test_data)
    if norm == 2:
        scaler = MinMaxScaler()
        scaler.fit(Train_data)
        Train_data = scaler.transform(Train_data)
        Test_data = scaler.transform(Test_data)

    data_new[:, 0:feature] = Train_data
    data_new[:, feature] = Train_label
    test_new[:,0:feature] = Test_data
    test_new[:,feature] = Test_label

    np.savetxt('datanewtrain.csv', data_new, delimiter=',')
    np.savetxt('datanewtest.csv', test_new, delimiter=',')

    #balance the data
    label1 = 0
    label2 = 0
    train_num = data_new.shape[0]
    for j in range(1, train_num):
        if data_new[j, feature] == 0:
            label1 = label1 + 1
        else:
            label2 = label2 + 1
    ratio = int(np.ceil(label1 / label2))
    count2 = 0
    B_Trainset = np.zeros(((ratio-1)*label2, feature))
    for i in range(0, train_num):
        if data_new[i, feature] == 1:
            for c in range(0, ratio-1):
                B_Trainset[count2+c, :] = data_new[i, 0:feature]
            count2 = count2 + ratio-2
    B_Trainset = B_Trainset[[i for i, x in enumerate(B_Trainset) if x.any()]]
    cut = B_Trainset.shape[0]
    dev = []
    for e in range(0, feature):
        dev.append(np.std(B_Trainset[:, e]))

    noisy = np.zeros((cut, feature))

    for b in range(0, feature):
        for c in range(0, cut):
            noisy[c,b] = np.random.uniform(-0.1*dev[b], 0.1*dev[b])

    B_Trainset = B_Trainset+noisy
    B_data = np.zeros((cut, feature+1))
    B_data[:,0:feature]=B_Trainset
    B_data[:,feature] = 1
    datab = np.vstack((data_new,B_data))


    # shuffle the data
    datab = shuffle(datab)

    TL = datab[:, feature]
    TD = datab[:, 0:feature]

# Outlier Detection
    train_num = TD.shape[0]
    LOF = LocalOutlierFactor(n_neighbors=80)
    Outlier = LOF.fit_predict(TD, TL)
    Train = np.zeros((train_num, feature))
    Tlabel = np.zeros(train_num)
    count3 = 0
    for c in range(0, train_num):
        if Outlier[c] == 1:
            Train[count3, :] = TD[c, :]
            Tlabel[count3] = TL[c]
            count3 = count3 + 1
    Train = Train[[i for i, x in enumerate(Trainset) if x.any()]]
    Tlabel = Tlabel[[i for i, x in enumerate(Trainset) if x.any()]]

    np.savetxt('B_Trainset_data.csv', Train, delimiter=',')
    np.savetxt('B_Trainset_label.csv', Tlabel, delimiter=',')
    np.savetxt('Test_data.csv', Test_data, delimiter=',')
    np.savetxt('Test_label.csv', Test_label, delimiter=',')
