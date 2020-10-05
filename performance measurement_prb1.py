import pandas as pd

data1 = pd.read_excel('D:/2/result10.xlsx')

y_pred = pd.DataFrame(data1,columns=['y_pred'])
#print(y_pred)
y = pd.DataFrame(data1,columns=['y'])
#print(y)

from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score


print("f1score = ", f1_score(y,y_pred))
print("precision = ", precision_score(y,y_pred))
print("recall = " , recall_score(y,y_pred))
