# Pre-processing
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import f_classif

def pre_select_data(selection,norm):
    Trainset = np.loadtxt('Trainset.csv', delimiter=',')
    (train_num, b) = Trainset.shape
    feature = b - 1
    Test = np.loadtxt('Test.csv', delimiter=',')
    test_num = Test.shape[0]
    Train_label = Trainset[:, feature]
    Train_info = Trainset[:, 0:feature]
    Test_info = Test[:, 0:feature]

    if selection == 1:
        fs = mutual_info_classif(X=Trainset, y=Train_label, random_state=1)
        count3 = 0
        for i in range(0, feature):
            if fs[i] == 0:
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
        (us,fs) = f_classif(X=Trainset, y=Train_label)
        count3 = 0
        for i in range(0, feature):
            if fs[i] >= 0.05:
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
        print('feature = ', feature)
    if selection == 0:
        feature = b - 1
        data_new = Trainset
        test_new = Test

# count for ratio and weight sample
    label1 = 0
    label2 = 0
    for j in range(1, train_num):
        if data_new[j, feature] == 0:
            label1 = label1 + 1
        else:
            label2 = label2 + 1
    ratio = label1 / label2

    Train_label = data_new[:, feature]
    Train_data = data_new[:, 0:feature]
    Test_label = test_new[:, feature]
    Test_data = test_new[:, 0:feature]

    np.savetxt('Bayes_label.csv', Train_label, delimiter=',')
    np.savetxt('Bayes.csv', Train_data, delimiter=',')
    np.savetxt('BayesTest.csv', Test_data, delimiter=',')
    np.savetxt('Bayes_TL.csv', Test_label, delimiter=',')

# Outlier Detection
    train_num = Train_data.shape[0]
    LOF = LocalOutlierFactor(n_neighbors=80)
    Outlier = LOF.fit_predict(Train_data, Train_label)
    Train = np.zeros((train_num, feature))
    Tlabel = np.zeros(train_num)
    count3 = 0
    for c in range(0, train_num):
        if Outlier[c] == 1:
            Train[count3, :] = Train_data[c, :]
            Tlabel[count3] = Train_label[c]
            count3 = count3 + 1
    Train = Train[[i for i, x in enumerate(Trainset) if x.any()]]
    Tlabel = Tlabel[[i for i, x in enumerate(Trainset) if x.any()]]

    train_num = Train.shape[0]
    sw = np.zeros(train_num)
    for j in range(0, train_num):
        if Tlabel[j] == 0:
            sw[j] = 1
        else:
            sw[j] = ratio

    test_num = Test_data.shape[0]

    if norm == 1:
        scaler = StandardScaler()
        scaler.fit(Train)
        Train = scaler.transform(Train)
        Test_data = scaler.transform(Test_data)
    if norm == 2:
        scaler = MinMaxScaler()
        scaler.fit(Train)
        Train = scaler.transform(Train)
        Test_data = scaler.transform(Test_data)

    np.savetxt('B_Trainset_data.csv', Train, delimiter=',')
    np.savetxt('B_Trainset_label.csv', Tlabel, delimiter=',')
    np.savetxt('Test_data.csv', Test_data, delimiter=',')
    np.savetxt('Test_label.csv', Test_label, delimiter=',')

    return(ratio,sw)

