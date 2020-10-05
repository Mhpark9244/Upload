import numpy as np
import pandas as pd
import random
from keras.layers import Dense
from keras.optimizers import  Adam
from keras.optimizers import  rmsprop
from keras.models import  Sequential


EPISODES = 2500

data = pd.read_csv("D:/Distancedata.csv")
data1 = pd.DataFrame(data,
                     columns=["Chip","Leadframe","Distance"],)
"""
===============================================
각각의 정보 담기
===============================================
"""
Chip = data1["Chip"]
Leadframe = data1["Leadframe"]
Distance = data1["Distance"]

Chip_list = list(set(Chip.index))
Leadframe_list = list(set(Leadframe.index))

"""
===============================================
 상태가 입력 큐함수가 출력인 인공신경망 생성
===============================================
"""
def build_model(self):
    model = Sequential()
    model.add(Dense(30, input_dim=self.state_size, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(self.action_size, activation='linear'))
    model.summary()
    model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
    return model

"""
===============================================
웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 1 -> leadframe 1
===============================================
"""


die1 = random.choice(Chip_list)

#print(type(Chip))
a = data1.ix[die1]
#print(a)
Chipno_1 = data1.ix[die1, 'Chip']
Leadframeno_1 = data1. ix[die1, 'Leadframe']
dist1 = data1.ix[die1, 'Distance']

print(" 1번째 시행 : ", "ChipNo_1 = ", Chipno_1,"\t","LeadframeNo_1= ", Leadframeno_1 +1 , "\t","Dist1 = ",dist1,"\t" )

#print(" 1번째 시행 : ","엑셀에서 row값은 =  ", die1 , "Chipno_1 = ", Chipno_1,"Leadframeno_1= ", Leadframeno_1 +1 , "Dist1 = ",dist1 )

Chip_list = Chip_list.remove(die1)

"""
===============================================
웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

#Leadframe 1 -> die 2
===============================================
"""

k = data1.ix[Leadframeno_1]
#print(k)
Leadframeno_1 = data1. ix[die1, 'Leadframe']

b = data1.ix[Leadframeno_1]
#print(b)

dist2 = data1.ix[Leadframeno_1, "Distance"]
#print(dist2)

Chipno_2 = data1.ix[Leadframeno_1, "Chip"]
#print("Chipno_2 =",  Chipno_2)

distance_sum = 0
distance_sum = dist1 + dist2
#print(distance_sum)

print(" 2번째 시행 : ","ChipNo_2 = ", Chipno_2,"\t","LeadframeNo_2= ", Leadframeno_1 +1 , "\t","Dist2 = ",dist2,"\t", "Distance_sum = ", distance_sum )
#print(" 2번째 시행 : ","엑셀에서 row값은 =  ", die1 ,"Chipno_2 = ", Chipno_2,"Leadframeno_1= ", Leadframeno_1 +1 , "Dist2 = ",dist2, "Distance_sum = ", distance_sum )

