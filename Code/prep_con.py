# Pre-processing
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import Imputer

def prep_conti():
    df = pd.read_csv('bank-additional.csv', delimiter=',')
    newdata = np.zeros((4119, 20))
    for i in range(0, 4119):
        newdata[i - 1, 0] = df.iloc[i, 0]
        if df.iloc[i, 1] == 'unknown':
            newdata[i - 1, 1] = 'NaN'
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
            newdata[i - 1, 2] = 11
        if df.iloc[i, 2] == 'unknown':
            newdata[i - 1, 2] = 'NaN'
        if df.iloc[i, 2] == 'married':
            newdata[i - 1, 2] = 1
        if df.iloc[i, 2] == 'single':
            newdata[i - 1, 2] = 2
        if df.iloc[i, 2] == 'divorced':
            newdata[i - 1, 2] = 3
        if df.iloc[i, 3] == 'unknown':
            newdata[i - 1, 3] = 'NaN'
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
            newdata[i - 1, 4] = 'NaN'
        if df.iloc[i, 4] == 'no':
            newdata[i - 1, 4] = 1
        if df.iloc[i, 4] == 'yes':
            newdata[i - 1, 4] = 2
        if df.iloc[i, 5] == 'unknown':
            newdata[i - 1, 5] = 'NaN'
        if df.iloc[i, 5] == 'no':
            newdata[i - 1, 5] = 1
        if df.iloc[i, 5] == 'yes':
            newdata[i - 1, 5] = 2
        if df.iloc[i, 6] == 'unknown':
            newdata[i - 1, 6] = 'NaN'
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
            newdata[i - 1, 13] = 'NaN'
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

    data = newdata[:,0:19]
    label = newdata[:,19]

    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    data_full = imp.fit_transform(data,label)
    newdata[:, 0:19] = data_full
    newdata[:, 19] = label

    np.savetxt('new.csv', newdata, delimiter=',')
