# Main program
import prep_FE
import Bayes
import Random_Forest
import LogicalRegression
import Boost
import prep_con
import KNN
import Prob_Con
import SVMC
import SGD
import numpy as np

print('----------------------------------------------------------------------------------------------')
print('EE559, Professor: Keith Jenkins')
print('Authors: Jinpeng Qi & Shuang Yuan')
print('Version: 2.0')
print('Including: Pre processing, Feature-space dimensionality adjustment, Cross validation, ')
print('Training and classification, Results displaying')
A = 0
print('----------------------------------------------------------------------------------------------')
print('Choose your pre-processing method:')
print('Warning: if some errors show in this part, please re-run the program again. Because Training set missing some certain feature by randomlizaton.')
print('1 -----> Continuous')
print('2 -----> Feature Expansion')
print('3 -----> Probability')
B = input('Method number = ')
if B == '1':
    data_set = prep_con.prep_conti()
    norm = 1
if B == '2':
    data_set = prep_FE.prepdata()
    norm = 2
if B == '3':
    data_set = Prob_Con.prob()
    norm = 3
print('----------------------------------------------------------------------------------------------')
print('Choose your feature selection method:')
print('0 -----> default no feature selection')
print('1 -----> mutual info')
print('2 -----> ExtraTreeBasis')
print('3------> F_value classif')
D = input('Method number = ')
selection = 0
if D == '1':
    selection = 1
if D == '2':
    selection = 2
if D == '3':
    selection = 3
print('----------------------------------------------------------------------------------------------')
print('Choose your imbalance method:')
print('1 -----> Random noise')
print('2 -----> weighted class')
C = input('Method number = ')
if C == '1':
    method = 1
if C == '2':
    method = 2
while A == 0:
    print('----------------------------------------------------------------------------------------------')
    print('choose your classification method:')
    print('1 -----> Bayes')
    print('2 -----> Random Forest')
    print('3 -----> Logical Regression')
    print('4 -----> Boost')
    print('5 -----> KNN')
    print('6 -----> SVM')
    print('7 -----> SGD')
    print('0 -----> exit this program')
    M = input('Method number = ')
    if M == '1':
        Bayes.GBC(method,norm,selection)
    if M == '2':
        Random_Forest.RandomF(method,norm,selection)
    if M == '3':
        LogicalRegression.LogicR(method,norm,selection)
    if M == '4':
        Boost.Ada(method,norm,selection)
    if M == '5':
        KNN.KClassifier(method,norm,selection)
    if M == '6':
        SVMC.support(method,norm,selection)
    if M == '7':
        SGD.SGD(method,norm,selection)
    if M == '0':
        print('The classification progress finished.')
        break
