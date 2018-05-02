# Pre-processing
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import shuffle
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import f_classif

from sklearn.preprocessing import scale

def pre_select_data(dataset,norm,selection):

    [a , b] = dataset.shape
    label_raw = dataset[:, b-1]
    data_raw = dataset[:, 0:b-1]
    if selection == 1:
        fs = mutual_info_classif(X=data_raw, y=label_raw, random_state=1)
        count3 = 0
        for i in range(0, 62):
            if fs[i] == 0:
                count3 = count3 + 1
        data_new = np.zeros((a, 63 - count3))
        count4 = 0
        for i in range(0, 62):
            if fs[i] != 0:
                data_new[:, count4] = dataset[:, i]
                count4 = count4 + 1
        feature = count4
        data_new[:, feature] = label_raw
        print('feature = ', feature)
    if selection == 2:
        clf = ExtraTreesClassifier()
        clf = clf.fit(data_raw, label_raw)
        model = SelectFromModel(clf, prefit=True)
        data_raw = model.transform(data_raw)
        (data_num,feature) = data_raw.shape
        data_new = np.zeros((data_num,feature+1))
        data_new[:,0:feature] = data_raw
        data_new[:,feature] = label_raw
        print('feature = ',feature)
    if selection == 3:
        (us,fs) = f_classif(X=data_raw, y=label_raw)
        count3 = 0
        (data_num,feature) = data_raw.shape
        for i in range(0, feature):
            if fs[i] >= 0.05:
                count3 = count3 + 1
        data_new = np.zeros((data_num, b - count3))
        count4 = 0
        for i in range(0, feature):
            if fs[i] < 0.05:
                data_new[:, count4] = data_raw[:, i]
                count4 = count4 + 1
        feature = count4
        data_new[:, feature] = label_raw
        print('feature = ', feature)
    if selection == 0:
        feature = b - 1
        data_new = np.zeros((a, feature + 1))
        data_new[:, 0:feature] = data_raw
        data_new[:, feature] = label_raw


    # select the trainset and testset which based on 15% dataset
    Random = np.random.randint(4120, size=618)
    Random = np.unique(Random)
    m = Random.shape[0]
    Test = np.zeros((m, feature + 1))
    Trainset = np.zeros((4119, feature + 1))
    for j in range(0, m - 1):
        Test[j, :] = data_new[Random[j], :]
    Test = Test[[i for i, x in enumerate(Test) if x.any()]]
    m = Test.shape[0]

    count1 = 0
    for k in range(0, 4119):
        flag = 0
        for n in range(1, m):
            if k == Random[n]:
                flag = 1
        if flag != 1:
            Trainset[int(count1), :] = data_new[int(k), :]
            count1 = count1 + 1

    # enumerate the O
    Trainset = Trainset[[i for i, x in enumerate(Trainset) if x.any()]]
    train_num = Trainset.shape[0]
    test_num = Test.shape[0]

    Trainset = shuffle(Trainset)
    Test = shuffle(Test)

    Train_data = Trainset[:, 0:feature]
    Train_label = Trainset[:, feature]
    Test_data = Test[:, 0:feature]
    Test_label = Test[:,feature]

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

    Trainset[:,0:feature] = Train_data
    Trainset[:,feature] = Train_label

    # balance the data
    label1 = 0
    label2 = 0
    train_num = Trainset.shape[0]
    for j in range(1, train_num):
        if Trainset[j, feature] == 0:
            label1 = label1 + 1
        else:
            label2 = label2 + 1
    ratio = int(np.ceil(label1 / label2))
    count2 = 0
    B_Trainset = np.zeros(((ratio-1)*label2, feature))
    for i in range(0, train_num):
        if Trainset[i, feature] == 1:
            for c in range(0, ratio-1):
                B_Trainset[count2+c, :] = Trainset[i, 0:feature]
            count2 = count2 + ratio-2
    B_Trainset = B_Trainset[[i for i, x in enumerate(B_Trainset) if x.any()]]
    cut = B_Trainset.shape[0]
    dev = []
    for e in range(0, feature):
        dev.append(np.std(B_Trainset[:, e]))

    noisy = np.zeros((cut, feature))

    for b in range(0,feature):
        for c in range(0,cut):
            noisy[c,b] = np.random.uniform(-0.1*dev[b], 0.1*dev[b])

    B_Trainset = B_Trainset+noisy
    B_data = np.zeros((cut, feature+1))
    B_data[:,0:feature]=B_Trainset
    B_data[:,feature] = 1
    datab = np.vstack((Trainset,B_data))

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
