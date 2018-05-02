# Cross Validation KNN
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import f1_score,roc_auc_score,roc_curve
from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import Axes3D

train_data_raw = np.loadtxt('B_Trainset_data.csv', delimiter=',')
test_data_raw = np.loadtxt('Test_data.csv', delimiter=',')
label_train = np.loadtxt('B_Trainset_label.csv', delimiter=',')
label_test = np.loadtxt('Test_label.csv', delimiter=',')

N = np.random.randint(5,120,size=30)

s = 30
FS = np.zeros(s)
DEV = np.zeros(s)

save_avg_acc = []

for j in range(0, 30):
    N_est = N[j]
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    all_fs = []
    for train_index, dev_index in skf.split(train_data_raw, label_train):
        X_cv_train, X_cv_dev = train_data_raw[train_index], train_data_raw[dev_index]
        y_cv_train, y_cv_dev = label_train[train_index], label_train[dev_index]
        clf = KNeighborsClassifier(n_neighbors=N_est)
        clf.fit(X_cv_train, y_cv_train)
        y_pred = clf.predict(X_cv_dev)
        f1 = f1_score(y_cv_dev, y_pred, average='weighted')
        all_fs.append(f1)
    mean_ = np.mean(all_fs)
    std_ = np.std(all_fs)
    FS[j] = mean_
    DEV[j] = std_

index = np.argmax(FS)
print('Best F1_Score = ', FS[index])
print('Stand deviation = ', DEV[index])
print('Best N_estimator = ', N[index])
N1 = N[index]


clf1 = KNeighborsClassifier(n_neighbors=N1)
clf1.fit(train_data_raw, label_train)
y_pred_train = clf1.predict(train_data_raw)
acc_train = accuracy_score(y_pred_train, label_train)
print('acc_train = ',acc_train)
target = ['class 1', 'class 2']
print(classification_report(label_train, y_pred_train, target_names=target))
print('auc = ', roc_auc_score(label_train, y_pred_train, average='weighted'))
y_pred_test = clf1.predict(test_data_raw)
acc_test = accuracy_score(y_pred_test, label_test)
print('acc_test = ',acc_test)
target = ['class 1', 'class 2']
print(classification_report(label_test, y_pred_test, target_names=target))
f1 = f1_score(label_test, y_pred_test, average='weighted')
print('f1 score =', f1)
auc = roc_auc_score(label_test, y_pred_test)
print('auc = ', auc)
fpr, tpr, thresholds = roc_curve(label_test, y_pred_test)
plt.plot(fpr, tpr)
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()

