# Pre-processing
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import Imputer
import cnt
from sklearn.utils import shuffle

def prob():
    df = pd.read_csv('bank-additional.csv', delimiter=',')

    newdata = np.zeros((4119, 20))
    for i in range(1, 4119):
        newdata[i - 1, 0] = df.iloc[i, 0]
        if df.iloc[i, 1] == 'unknown':
            newdata[i - 1, 1] = 0
        if df.iloc[i, 1] == 'admin.':
            newdata[i - 1, 1] = 1
        if df.iloc[i, 1] == 'blue-collar':
            newdata[i - 1, 1] = 2
        if df.iloc[i, 1] == 'entrepreneur':
            newdata[i - 1, 1] = 3
        if df.iloc[i, 1] == 'housemaid':
            newdata[i - 1, 1] = 4
        if df.iloc[i, 1] == 'management':
            newdata[i - 1, 1] = 5
        if df.iloc[i, 1] == 'retired':
            newdata[i - 1, 1] = 6
        if df.iloc[i, 1] == 'self-employed':
            newdata[i - 1, 1] = 7
        if df.iloc[i, 1] == 'services':
            newdata[i - 1, 1] = 8
        if df.iloc[i, 1] == 'student':
            newdata[i - 1, 1] = 9
        if df.iloc[i, 1] == 'technician':
            newdata[i - 1, 1] = 10
        if df.iloc[i, 1] == 'unemployed':
            newdata[i - 1, 1] = 11
        if df.iloc[i, 2] == 'unknown':
            newdata[i - 1, 2] = 0
        if df.iloc[i, 2] == 'married':
            newdata[i - 1, 2] = 1
        if df.iloc[i, 2] == 'single':
            newdata[i - 1, 2] = 2
        if df.iloc[i, 2] == 'divorced':
            newdata[i - 1, 2] = 3
        if df.iloc[i, 3] == 'unknown':
            newdata[i - 1, 3] = 0
        if df.iloc[i, 3] == 'basic.4y':
            newdata[i - 1, 3] = 1
        if df.iloc[i, 3] == 'basic.6y':
            newdata[i - 1, 3] = 2
        if df.iloc[i, 3] == 'basic.9y':
            newdata[i - 1, 3] = 3
        if df.iloc[i, 3] == 'high.school':
            newdata[i - 1, 3] = 4
        if df.iloc[i, 3] == 'illiterate':
            newdata[i - 1, 3] = 5
        if df.iloc[i, 3] == 'professional.course':
            newdata[i - 1, 3] = 6
        if df.iloc[i, 3] == 'university.degree':
            newdata[i - 1, 3] = 7
        if df.iloc[i, 4] == 'unknown':
            newdata[i - 1, 4] = 0
        if df.iloc[i, 4] == 'no':
            newdata[i - 1, 4] = 1
        if df.iloc[i, 4] == 'yes':
            newdata[i - 1, 4] = 2
        if df.iloc[i, 5] == 'unknown':
            newdata[i - 1, 5] = 0
        if df.iloc[i, 5] == 'no':
            newdata[i - 1, 5] = 1
        if df.iloc[i, 5] == 'yes':
            newdata[i - 1, 5] = 2
        if df.iloc[i, 6] == 'unknown':
            newdata[i - 1, 6] = 0
        if df.iloc[i, 6] == 'no':
            newdata[i - 1, 6] = 1
        if df.iloc[i, 6] == 'yes':
            newdata[i - 1, 6] = 2
        if df.iloc[i, 7] == 'cellular':
            newdata[i - 1, 7] = 1
        if df.iloc[i, 7] == 'telephone':
            newdata[i - 1, 7] = 2
        if df.iloc[i, 8] == 'apr':
            newdata[i - 1, 8] = 1
        if df.iloc[i, 8] == 'aug':
            newdata[i - 1, 8] = 2
        if df.iloc[i, 8] == 'dec':
            newdata[i - 1, 8] = 3
        if df.iloc[i, 8] == 'jul':
            newdata[i - 1, 8] = 4
        if df.iloc[i, 8] == 'jun':
            newdata[i - 1, 8] = 5
        if df.iloc[i, 8] == 'mar':
            newdata[i - 1, 8] = 6
        if df.iloc[i, 8] == 'may':
            newdata[i - 1, 8] = 7
        if df.iloc[i, 8] == 'nov':
            newdata[i - 1, 8] = 8
        if df.iloc[i, 8] == 'oct':
            newdata[i - 1, 8] = 9
        if df.iloc[i, 8] == 'sep':
            newdata[i - 1, 8] = 10
        if df.iloc[i, 9] == 'fri':
            newdata[i - 1, 9] = 1
        if df.iloc[i, 9] == 'mon':
            newdata[i - 1, 9] = 2
        if df.iloc[i, 9] == 'thu':
            newdata[i - 1, 9] = 3
        if df.iloc[i, 9] == 'tue':
            newdata[i - 1, 9] = 4
        if df.iloc[i, 9] == 'wed':
            newdata[i - 1, 9] = 5
        newdata[i - 1, 10] = df.iloc[i, 10]
        newdata[i - 1, 11] = df.iloc[i, 11]
        newdata[i - 1, 12] = df.iloc[i, 12]
        if df.iloc[i, 13] == 'nonexistent':
            newdata[i - 1, 13] = 0
        if df.iloc[i, 13] == 'failure':
            newdata[i - 1, 13] = 1
        if df.iloc[i, 13] == 'success':
            newdata[i - 1, 13] = 2
        newdata[i - 1, 14] = df.iloc[i, 14]
        newdata[i - 1, 15] = df.iloc[i, 15]
        newdata[i - 1, 16] = df.iloc[i, 16]
        newdata[i - 1, 17] = df.iloc[i, 17]
        newdata[i - 1, 18] = df.iloc[i, 18]
        if df.iloc[i, 19] == 'no':
            newdata[i - 1, 19] = 0
        else:
            newdata[i - 1, 19] = 1

    np.savetxt('see.csv', newdata, delimiter=',')

    #select the test
    Random = np.random.randint(4120, size=618)
    Random = np.unique(Random)
    m = Random.shape[0]
    Test = np.zeros((m, 20))
    Trainset = np.zeros((4119, 20))

    for j in range(0, m - 1):
        Test[j, :] = newdata[Random[j], :]
    Test = Test[[i for i, x in enumerate(Test) if x.any()]]
    test_num = Test.shape[0]

    count1 = 0
    for k in range(0, 4119):
        flag = 0
        for n in range(1, m):
            if k == Random[n]:
                flag = 1
        if flag != 1:
            Trainset[count1, :] = newdata[k, :]
            count1 = count1 + 1

    # enumerate the O
    Trainset = Trainset[[i for i, x in enumerate(Trainset) if x.any()]]
    train_num = Trainset.shape[0]

    label = newdata[:,19]
    a = cnt.CNT(label, 0)
    b = cnt.CNT(label, 1)

    label1 = np.zeros((a, 20))
    label2 = np.zeros((b, 20))
    ratio = int(np.fix(a / b))
    count1 = 0
    count2 = 0

    for j in range(0, 4119):
        if newdata[j, 19] == 0:
            label1[count1, :] = newdata[j, :]
            count1 = count1 + 1
        if newdata[j, 19] == 1:
            label2[count2, :] = newdata[j, :]
            count2 = count2 + 1

    for k in range(0, 12):
        L1 = cnt.CNT(Trainset[:, 1], k)
        L2 = cnt.CNT(label2[:,1], k)
        Trainset[:, 1] = np.where(Trainset[:, 1] == k, L2 / (L1+L2) , Trainset[:, 1])
        Test[:, 1] = np.where(Test[:, 1] == k, L2 / (L1+L2), Test[:, 1])

    for k in range(0, 4):
        L1 = cnt.CNT(Trainset[:, 2], k)
        L2 = cnt.CNT(label2[:, 2], k)
        Trainset[:, 2] = np.where(Trainset[:, 2] == k, L2 / (L1+L2), Trainset[:, 2])
        Test[:, 2] = np.where(Test[:, 2] == k, L2 / (L1+L2), Test[:, 2])

    for k in range(0, 8):
        L1 = cnt.CNT(Trainset[:, 3], k)
        L2 = cnt.CNT(label2[:, 3], k)
        Trainset[:, 3] = np.where(Trainset[:, 3] == k, L2 / (L1+L2), Trainset[:, 3])
        Test[:, 3] = np.where(Test[:, 3] == k, L2 / (L1+L2), Test[:, 3])

    for k in range(0, 3):
        L1 = cnt.CNT(Trainset[:, 4], k)
        L2 = cnt.CNT(label2[:, 4], k)
        Trainset[:, 4] = np.where(Trainset[:, 4] == k, L2 / (L1+L2), Trainset[:, 4])
        Test[:, 4] = np.where(Test[:, 4] == k, L2 / (L1+L2), Test[:, 4])

    for k in range(0, 3):
        L1 = cnt.CNT(Trainset[:, 5], k)
        L2 = cnt.CNT(label2[:, 5], k)
        Trainset[:, 5] = np.where(Trainset[:, 5] == k, L2 / (L1+L2), Trainset[:, 5])
        Test[:, 5] = np.where(Test[:, 5] == k, L2 / (L1+L2), Test[:, 5])

    for k in range(0, 3):
        L1 = cnt.CNT(Trainset[:, 6], k)
        L2 = cnt.CNT(label2[:, 6], k)
        Trainset[:, 6] = np.where(Trainset[:, 6] == k, L2 / (L1+L2), Trainset[:, 6])
        Test[:, 6] = np.where(Test[:, 6] == k, L2 / (L1+L2), Test[:, 6])

    for k in range(1, 3):
        L1 = cnt.CNT(Trainset[:, 7], k)
        L2 = cnt.CNT(label2[:, 7], k)
        Trainset[:, 7] = np.where(Trainset[:, 7] == k, L2 / (L1+L2), Trainset[:, 7])
        Test[:, 7] = np.where(Test[:, 7] == k, L2 / (L1+L2), Test[:, 7])

    for k in range(1, 11):
        L1 = cnt.CNT(Trainset[:, 8], k)
        L2 = cnt.CNT(label2[:, 8], k)
        Trainset[:, 8] = np.where(Trainset[:, 8] == k, L1 / (L1+L2), Trainset[:, 8])
        Test[:, 8] = np.where(Test[:, 8] == k, L1 / (L1+L2), Test[:, 8])

    for k in range(1, 6):
        L1 = cnt.CNT(Trainset[:, 9], k)
        L2 = cnt.CNT(label2[:, 9], k)
        Trainset[:, 9] = np.where(Trainset[:, 9] == k, L2 / (L1+L2), Trainset[:, 9])
        Test[:, 9] = np.where(Test[:, 9] == k, L2 / (L1+L2), Test[:, 9])

    for k in range(0, 3):
        L1 = cnt.CNT(Trainset[:, 13], k)
        L2 = cnt.CNT(label2[:, 13], k)
        Trainset[:, 13] = np.where(Trainset[:, 13] == k, L2 / (L1+L2), Trainset[:, 13])
        Test[:, 13] = np.where(Test[:, 13] == k, L2 / (L1+L2), Test[:, 13])

    np.savetxt('Trainset.csv', Trainset, delimiter=',')
    np.savetxt('Test.csv', Test, delimiter=',')
