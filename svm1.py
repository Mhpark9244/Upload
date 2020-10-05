"""
#예제 선형SVM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

bankdata = pd.read_csv('D:/bill_authentication.csv')
#print(bankdata)

X = bankdata.drop('Class',axis=1)
y = bankdata['Class']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.20)

from sklearn.svm import SVC # SVM 모델에서 SVC 방법 적용
svclassifier = SVC(kernel='linear') # kernel은 선형(linear) 사용
svclassifier.fit(X_train,y_train) # 분류기에 training set을 fitting

y_pred = svclassifier.predict(X_test) # 예측

from sklearn.metrics import classification_report, confusion_matrix #모델 평가model evaluation
print(confusion_matrix(y_test, y_pred)) # 혼동행렬로 결과 출력
print(classification_report(y_test, y_pred)) # 결과 지표 출력
"""
# 예제 비선형 & 여러가지 커널 사용
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
irisdata = load_iris()
print(irisdata)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

colnames = ['sepal_length','sepal_width','petal_length','petal_width','Class'] # 데이터 불러오기, 열 이름과 클래스 붙여주기
irisdata = pd.read_csv(url, names=colnames)
X = irisdata.drop('Class', axis = 1)
y = irisdata['Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

#polynomial kernel 사용
from sklearn.svm import SVC
svclassifier = SVC(kernel='poly', degree = 8)
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#gaussian kernel 사용
from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Sigmiod kernel 사용
from sklearn.svm import SVC
svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

