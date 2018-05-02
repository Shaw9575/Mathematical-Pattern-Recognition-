# Pre-processing
import numpy as np
import pandas as pd
from scipy import stats

def prepdata():
    df = pd.read_csv('bank-additional.csv', delimiter=',')

    df.insert(2, 'admin.', df['job'])
    df.insert(3, 'blue_collar', df['job'])
    df.insert(4, 'entrepreneur', df['job'])
    df.insert(5, 'housemaid', df['job'])
    df.insert(6, 'management', df['job'])
    df.insert(7, 'retired', df['job'])
    df.insert(8, 'self-employed', df['job'])
    df.insert(9, 'services', df['job'])
    df.insert(10, 'student', df['job'])
    df.insert(11, 'technician', df['job'])
    df.insert(12, 'unemployed', df['job'])

    df.insert(14, 'married', df['marital'])
    df.insert(15, 'single', df['marital'])
    df.insert(16, 'divorced', df['marital'])

    df.insert(18, 'basic.4y', df['education'])
    df.insert(19, 'basic.6y', df['education'])
    df.insert(20, 'basic.9y', df['education'])
    df.insert(21, 'high.school', df['education'])
    df.insert(22, 'illiterate', df['education'])
    df.insert(23, 'professional.course', df['education'])
    df.insert(24, 'university.degree', df['education'])

    df.insert(26, 'no_d', df['default'])
    df.insert(27, 'yes_d', df['default'])

    df.insert(29, 'no_h', df['housing'])
    df.insert(30, 'yes_h', df['housing'])

    df.insert(32, 'no_l', df['loan'])
    df.insert(33, 'yes_l', df['loan'])

    df.insert(35, 'cellular', df['contact'])

    df.insert(37, 'aug', df['month'])
    df.insert(38, 'dec', df['month'])
    df.insert(39, 'jul', df['month'])
    df.insert(40, 'jun', df['month'])
    df.insert(41, 'mar', df['month'])
    df.insert(42, 'may', df['month'])
    df.insert(43, 'nov', df['month'])
    df.insert(44, 'oct', df['month'])
    df.insert(45, 'sep', df['month'])

    df.insert(47, 'mon', df['day_of_week'])
    df.insert(48, 'wed', df['day_of_week'])
    df.insert(49, 'thu', df['day_of_week'])
    df.insert(50, 'tue', df['day_of_week'])

    df.insert(55, 'failure', df['poutcome'])
    df.insert(56, 'success', df['poutcome'])

    newdata = np.zeros((4119, 63))
    for i in range(0, 4119):
        newdata[i - 1, 0] = df.iloc[i, 0]
        if df.iloc[i, 1] == 'unknown':
            newdata[i - 1, 1] = 1
        if df.iloc[i, 2] == 'admin.':
            newdata[i - 1, 2] = 1
        if df.iloc[i, 3] == 'blue-collar':
            newdata[i - 1, 3] = 1
        if df.iloc[i, 4] == 'entrepreneur':
            newdata[i - 1, 4] = 1
        if df.iloc[i, 5] == 'housemaid':
            newdata[i - 1, 5] = 1
        if df.iloc[i, 6] == 'management':
            newdata[i - 1, 6] = 1
        if df.iloc[i, 7] == 'retired':
            newdata[i - 1, 7] = 1
        if df.iloc[i, 8] == 'self-employed':
            newdata[i - 1, 8] = 1
        if df.iloc[i, 9] == 'services':
            newdata[i - 1, 9] = 1
        if df.iloc[i, 10] == 'student':
            newdata[i - 1, 10] = 1
        if df.iloc[i, 11] == 'technician':
            newdata[i - 1, 11] = 1
        if df.iloc[i, 12] == 'unemployed':
            newdata[i - 1, 12] = 1
        if df.iloc[i, 13] == 'unknown':
            newdata[i - 1, 13] = 1
        if df.iloc[i, 14] == 'married':
            newdata[i - 1, 14] = 1
        if df.iloc[i, 15] == 'single':
            newdata[i - 1, 15] = 1
        if df.iloc[i, 16] == 'divorced':
            newdata[i - 1, 16] = 1
        if df.iloc[i, 17] == 'unknown':
            newdata[i - 1, 17] = 1
        if df.iloc[i, 18] == 'basic.4y':
            newdata[i - 1, 18] = 1
        if df.iloc[i, 19] == 'basic.6y':
            newdata[i - 1, 19] = 1
        if df.iloc[i, 20] == 'basic.9y':
            newdata[i - 1, 20] = 1
        if df.iloc[i, 21] == 'high.school':
            newdata[i - 1, 21] = 1
        if df.iloc[i, 22] == 'illiterate':
            newdata[i - 1, 22] = 1
        if df.iloc[i, 23] == 'professional.course':
            newdata[i - 1, 23] = 1
        if df.iloc[i, 24] == 'university.degree':
            newdata[i - 1, 24] = 1
        if df.iloc[i, 25] == 'unknown':
            newdata[i - 1, 25] = 1
        if df.iloc[i, 26] == 'no':
            newdata[i - 1, 26] = 1
        if df.iloc[i, 27] == 'yes':
            newdata[i - 1, 27] = 1
        if df.iloc[i, 28] == 'unknown':
            newdata[i - 1, 28] = 1
        if df.iloc[i, 29] == 'no':
            newdata[i - 1, 29] = 1
        if df.iloc[i, 30] == 'yes':
            newdata[i - 1, 30] = 1
        if df.iloc[i, 31] == 'unknown':
            newdata[i - 1, 31] = 1
        if df.iloc[i, 32] == 'no':
            newdata[i - 1, 32] = 1
        if df.iloc[i, 33] == 'yes':
            newdata[i - 1, 33] = 1
        if df.iloc[i, 34] == 'cellular':
            newdata[i - 1, 34] = 1
        if df.iloc[i, 35] == 'telephone':
            newdata[i - 1, 35] = 1
        if df.iloc[i, 36] == 'apr':
            newdata[i - 1, 36] = 1
        if df.iloc[i, 37] == 'aug':
            newdata[i - 1, 37] = 1
        if df.iloc[i, 38] == 'dec':
            newdata[i - 1, 38] = 1
        if df.iloc[i, 39] == 'jul':
            newdata[i - 1, 39] = 1
        if df.iloc[i, 40] == 'jun':
            newdata[i - 1, 40] = 1
        if df.iloc[i, 41] == 'mar':
            newdata[i - 1, 41] = 1
        if df.iloc[i, 42] == 'may':
            newdata[i - 1, 42] = 1
        if df.iloc[i, 43] == 'nov':
            newdata[i - 1, 43] = 1
        if df.iloc[i, 44] == 'oct':
            newdata[i - 1, 44] = 1
        if df.iloc[i, 45] == 'sep':
            newdata[i - 1, 45] = 1
        if df.iloc[i, 46] == 'fri':
            newdata[i - 1, 46] = 1
        if df.iloc[i, 47] == 'mon':
            newdata[i - 1, 47] = 1
        if df.iloc[i, 48] == 'thu':
            newdata[i - 1, 48] = 1
        if df.iloc[i, 49] == 'tue':
            newdata[i - 1, 49] = 1
        if df.iloc[i, 50] == 'wed':
            newdata[i - 1, 50] = 1
        newdata[i - 1, 51] = df.iloc[i, 51]
        newdata[i - 1, 52] = df.iloc[i, 52]
        newdata[i - 1, 53] = df.iloc[i, 53]
        if df.iloc[i, 54] == 'nonexistent':
            newdata[i - 1, 54] = 1
        if df.iloc[i, 55] == 'failure':
            newdata[i - 1, 55] = 1
        if df.iloc[i, 56] == 'success':
            newdata[i - 1, 56] = 1
        newdata[i - 1, 57] = df.iloc[i, 57]
        newdata[i - 1, 58] = df.iloc[i, 58]
        newdata[i - 1, 59] = df.iloc[i, 59]
        newdata[i - 1, 60] = df.iloc[i, 60]
        newdata[i - 1, 61] = df.iloc[i, 61]
        if df.iloc[i, 62] == 'no':
            newdata[i - 1, 62] = 0
        else:
            newdata[i - 1, 62] = 1

    # newdata[:, 0] = stats.zscore(newdata[:, 0])
    # newdata[:, 51] = stats.zscore(newdata[:, 51])
    # newdata[:, 52] = stats.zscore(newdata[:, 52])
    # newdata[:, 53] = stats.zscore(newdata[:, 53])
    # newdata[:, 57] = stats.zscore(newdata[:, 57])
    # newdata[:, 58] = stats.zscore(newdata[:, 58])
    # newdata[:, 59] = stats.zscore(newdata[:, 59])
    # newdata[:, 60] = stats.zscore(newdata[:, 60])
    # newdata[:, 61] = stats.zscore(newdata[:, 61])

    np.savetxt('new.csv', newdata, delimiter=',')
