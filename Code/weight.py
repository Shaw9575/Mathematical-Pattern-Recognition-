# Pre-processing
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

def pre_select_data(dataset,norm,selection):
    [a, b] = dataset.shape
    label_raw = dataset[:, b - 1]
    data_raw = dataset[:, 0:b - 1]
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
        print(clf.feature_importances_)
        model = SelectFromModel(clf, prefit=True)
        data_raw = model.transform(data_raw)
        (data_num,feature) = data_raw.shape
        data_new = np.zeros((data_num,feature+1))
        data_new[:,0:feature] = data_raw
        data_new[:,feature] = label_raw
        print('feature = ',feature)
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

    for j in range(0, m-1):
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
            Trainset[count1, :] = data_new[k, :]
            count1 = count1 + 1

    # enumerate the O
    Trainset = Trainset[[i for i, x in enumerate(Trainset) if x.any()]]
    train_num = Trainset.shape[0]
# count for ratio and weight sample
    label1 = 0
    label2 = 0
    for j in range(1, train_num):
        if Trainset[j, feature] == 0:
            label1 = label1 + 1
        else:
            label2 = label2 + 1
    ratio = label1 / label2

    Train_label = Trainset[:, feature]
    Train_data = Trainset[:, 0:feature]
    Test_label = Test[:, feature]
    Test_data = Test[:, 0:feature]

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
