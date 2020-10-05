import numpy as np
import pandas as pd
import random
from keras.layers import Dense
from keras.optimizers import  Adam
from keras.optimizers import  rmsprop
from keras.models import  Sequential
import tensorflow as tf
import sys

EPISODES = 2500

data = pd.read_csv("D:/Distancedata_uni90.csv")
data1 = pd.DataFrame(data,
                     columns=["Chip","Leadframe","Distance"],)
data2 = pd.DataFrame(data,
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
die_1 = random.randrange(0, 81)
die_1_info = data1[data1['Chip']  == die_1]


c = list(die_1_info.Distance)
dist_1_info = sorted(c)
dist_1_min = dist_1_info[0]
f1 = int(dist_1_min)


g = die_1_info[die_1_info.Distance == f1]
h = g.Leadframe                                                                                                  # 길이가 같아면 숫자가 작은 쪽을 선택
h1 = list(h)
Leadframe_1 = h1[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info                                                                                                     # i 는 거리 최소값 뺀 거리 리스트
                                                                                                                            # 모두 다 리스트 형태 자료형

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_1_info_m, count)
ran = map(int, ran)
for t1 in ran:
    t1 = t1
    #print(t1)

for i in range(1):
    epsilon = random.random()

if epsilon > 0.2:
    dist_1 = f1
else:
    dist_1 = t1
                                                                                                                            # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.

data1 = data1[data1.Chip != die_1]                             # 옮겨진die1에 해당하는 애는 삭제
data1 = data1

"""
===============================================
웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# leadframe 1 -> die 2
===============================================
"""
Leadframe_1_info = data1[data1['Leadframe']  == Leadframe_1]
#print(Leadframe1_info)

c2 = list(Leadframe_1_info.Distance)
dist_2_info = sorted(c2)


dist_2_info = set(dist_2_info)
#print(dist2_info)
dist_2_info = list(dist_2_info)
dist_2_info = sorted(dist_2_info)
#print("dist2_info =" , dist2_info)


dist_2_min = dist_2_info[0]
f2 = int(dist_2_min)
#print(f2)

g2 = Leadframe_1_info[Leadframe_1_info.Distance == f2]
#print("g2 = ", g2)

h2 = g2.Chip
#print(h2)
h3 = list(h2)
#print(h3)
die_2 = h3[0]                                                                                                    # 길이가 같아면 숫자가 작은 쪽을 선택
#print("Die2 = ", Die_2)

del dist_2_info[0]
dist_2_info_m= dist_2_info                                                                                                     # i 는 거리 최소값 뺀 거리 리스트
dist_2_info_m = set(dist_2_info_m)
dist_2_info_m = list(dist_2_info)
#print("dist2_info_m = ", dist2_info_m)                                                                                                    # 모두 다 리스트 형태 자료형

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_2_info_m, count)
#print(ran)
ran = map(int, ran)
for t2 in ran:
    t2 = t2
    #print(t2)

for i in range(1):
    epsilon = random.random()
   #print(epsilon)

if epsilon > 0.2:
    dist_2 = f2
    #print("dis_2 = " , dist_2)
else:
    dist_2 = t2
    #print("dis_2 = " ,dist_2)                                                                               # 여기까지가 옮겨진Leadframe_1에서Die_2까지 이동경로 구하기

data1 = data1[data1.Leadframe != Leadframe_1]      # 옮겨진Leadframe1에 해당하는 애는 삭제

#print(data1)

total_dist = dist_1 + dist_2
print("[1]", "die_1 = ", die_1, "Leadframe_1 = ", Leadframe_1, "die_2 = ", die_2, "\t","Dist = ", dist_1,"Dist = ", dist_2, "total_dist = ", total_dist)

"""
===============================================
[2] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 2 -> leadframe 2, leadframe 2 -> die 3 f,t,dist = 3,4
===============================================
"""
dienum= [ x for x in range(1,82)]
leadframenum = [x for x in range(1,41)]
num1 = [x for x  in range(200) if x % 2 ==1]
num2 = [x for x  in range(1, 200) if x % 2 ==0]

i = int(die_2)


die_info = data1[data1['Chip']  == i]
f =  int(sorted(list(die_info.Distance))[0])
dist_3_info = sorted(list(die_info.Distance))

Leadframe_2 = list(die_info[die_info.Distance == f].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

j = int(Leadframe_2)

del dist_3_info[0]
dist_3_info_m = dist_3_info

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_1_info_m, count)
ran = map(int, ran)
for t in ran:
    t = t
    #print(t3)

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_3 = f
    else:
        dist_3 = t                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1[data1.Chip != die_2]                                        # 옮겨진die1에 해당하는 애는 삭제
total_dist = dist_1 + dist_2 + dist_3

Leadframe_2_info = data1[data1['Leadframe']  == Leadframe_2]

dist_4_info = sorted(list(set(sorted(list(Leadframe_2_info.Distance)))))
dist_4_min = dist_4_info[0]
f4 = int(dist_4_min)
del dist_4_info[0]
dist_4_info_m = list(set(dist_4_info))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_3 = list(Leadframe_2_info[Leadframe_2_info.Distance == f4].Chip)[0]  # 길이가 같아면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_4_info_m, count)
ran = map(int, ran)
for t4 in ran:
    t4 = t4

for i in range(1):
    epsilon = random.random()
   #print(epsilon)

if epsilon > 0.2:
    dist_4 = f4
else:
    dist_4 = t4

data1 = data1[data1.Leadframe != Leadframe_2]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = dist_1 + dist_2 + dist_3+dist_4
print("[2]", "die_2 = ", die_2, "Leadframe_2 = ", Leadframe_2, "die_3 = ", die_3, "\t","Dist = ", dist_3,"Dist = ", dist_4, "total_dist = ", total_dist)
"""
===============================================
[3] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 3 (i) -> leadframe 3 , leadframe 3(j) -> die 4(i+1),,,, f= 5,6 t=5,6, dist = 5,6 /// 5 = l, 6 = k
===============================================
"""

f_5 =  int(sorted(list(data1[data1['Chip']  == die_3].Distance))[0])
dist_5_info = sorted(list(data1[data1['Chip']  == die_3].Distance))

Leadframe_3 = list(data1[data1['Chip']  == die_3][data1[data1['Chip']  == die_3].Distance == f_5].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_3].Distance))[0]
dist_5_info_m = sorted(list(data1[data1['Chip']  == die_3].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_5_info_m, count)
ran = map(int, ran)
for t_5 in ran:
    t_5 = t_5

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_5 = f_5
    else:
        dist_5 = t_5                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_3].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_6 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_3].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_3].Distance)))))[0]
dist_6_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_3].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_4 = list(data1[data1['Leadframe']  == Leadframe_3][data1[data1['Leadframe']  == Leadframe_3].Distance == f_6].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_6_info_m, count)
ran = map(int, ran)
for t_6 in ran:
    t_6 = t_6

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_6 = f_6
else:
    dist_6 = t_6

data1 = data1[data1.Leadframe != Leadframe_3]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,7):
    exec('total_dist = total_dist + dist_' + str(a))

print("[3]", "die_3 = ", die_3, "Leadframe_3 = ", Leadframe_3, "die_4 = ", die_4, "\t", "total_dist = ", total_dist)
"""
===============================================
[4] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 4 (i) -> leadframe 4 , leadframe 4(j) -> die 5(i+1),,,, f= 7 t=8, dist = 7,8 
===============================================
"""
f_7 =  int(sorted(list(data1[data1['Chip']  == die_4].Distance))[0])
dist_7_info = sorted(list(data1[data1['Chip']  == die_4].Distance))

Leadframe_4 = list(data1[data1['Chip']  == die_4][data1[data1['Chip']  == die_4].Distance == f_7].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_4].Distance))[0]
dist_7_info_m = sorted(list(data1[data1['Chip']  == die_4].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_7_info_m, count)
ran = map(int, ran)
for t_8 in ran:
    t_8 = t_8

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_7 = f_7
    else:
        dist_7 = t_8                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_4].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_6 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_4].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_4].Distance)))))[0]
dist_8_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_4].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_5 = list(data1[data1['Leadframe']  == Leadframe_4][data1[data1['Leadframe']  == Leadframe_4].Distance == f_6].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_8_info_m, count)
ran = map(int, ran)
for t_8 in ran:
    t_8 = t_8

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_8 = f_6
else:
    dist_8 = t_8

data1 = data1[data1.Leadframe != Leadframe_4]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,9):
    exec('total_dist = total_dist + dist_' + str(a))

print("[4]", "die_4 = ", die_4, "Leadframe_4 = ", Leadframe_4, "die_5 = ", die_5, "\t", "total_dist = ", total_dist)
"""
===============================================
[5] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_9 =  int(sorted(list(data1[data1['Chip']  == die_5].Distance))[0])
dist_9_info = sorted(list(data1[data1['Chip']  == die_5].Distance))

Leadframe_5 = list(data1[data1['Chip']  == die_5][data1[data1['Chip']  == die_5].Distance == f_9].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_5].Distance))[0]
dist_9_info_m = sorted(list(data1[data1['Chip']  == die_5].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_9_info_m, count)
ran = map(int, ran)
for t_10 in ran:
    t_10 = t_10

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_9 = f_9
    else:
        dist_9 = t_10                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_5].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_9 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_5].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_5].Distance)))))[0]
dist_10_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_5].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_6 = list(data1[data1['Leadframe']  == Leadframe_5][data1[data1['Leadframe']  == Leadframe_5].Distance == f_9].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_10_info_m, count)
ran = map(int, ran)
for t_10 in ran:
    t_10 = t_10

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_10 = f_6
else:
    dist_10 = t_10

data1 = data1[data1.Leadframe != Leadframe_5]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,11):
    exec('total_dist = total_dist + dist_' + str(a))

print("[5]", "die_5 = ", die_5, "Leadframe_5 = ", Leadframe_5, "die_6 = ", die_6, "\t", "total_dist = ", total_dist)
"""
===============================================
[6] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_11 =  int(sorted(list(data1[data1['Chip']  == die_6].Distance))[0])
dist_11_info = sorted(list(data1[data1['Chip']  == die_6].Distance))

Leadframe_6 = list(data1[data1['Chip']  == die_6][data1[data1['Chip']  == die_6].Distance == f_11].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_6].Distance))[0]
dist_11_info_m = sorted(list(data1[data1['Chip']  == die_6].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_11_info_m, count)
ran = map(int, ran)
for t_12 in ran:
    t_12 = t_12

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_11 = f_11
    else:
        dist_11 = t_12                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_6].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_11 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_6].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_6].Distance)))))[0]
dist_12_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_6].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_7 = list(data1[data1['Leadframe']  == Leadframe_6][data1[data1['Leadframe']  == Leadframe_6].Distance == f_11].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_12_info_m, count)
ran = map(int, ran)
for t_12 in ran:
    t_12 = t_12

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_12 = f_11
else:
    dist_12 = t_12

data1 = data1[data1.Leadframe != Leadframe_6]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,13):
    exec('total_dist = total_dist + dist_' + str(a))

print("[6]", "die_6 = ", die_6, "Leadframe_6 = ", Leadframe_6, "die_7 = ", die_7, "\t", "total_dist = ", total_dist)
"""
===============================================
[7] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_13 =  int(sorted(list(data1[data1['Chip']  == die_7].Distance))[0])
dist_13_info = sorted(list(data1[data1['Chip']  == die_7].Distance))

Leadframe_7 = list(data1[data1['Chip']  == die_7][data1[data1['Chip']  == die_7].Distance == f_13].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_7].Distance))[0]
dist_13_info_m = sorted(list(data1[data1['Chip']  == die_7].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_13_info_m, count)
ran = map(int, ran)
for t_14 in ran:
    t_14 = t_14

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_13 = f_13
    else:
        dist_13 = t_14                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_7].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_14 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_7].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_7].Distance)))))[0]
dist_14_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_7].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_8 = list(data1[data1['Leadframe']  == Leadframe_7][data1[data1['Leadframe']  == Leadframe_7].Distance == f_14].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_14_info_m, count)
ran = map(int, ran)
for t_14 in ran:
    t_14 = t_14

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_14 = f_14
else:
    dist_14 = t_14

data1 = data1[data1.Leadframe != Leadframe_7]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,15):
    exec('total_dist = total_dist + dist_' + str(a))

print("[7]", "die_7 = ", die_7, "Leadframe_7 = ", Leadframe_7, "die_8 = ", die_8, "\t", "total_dist = ", total_dist)
"""
===============================================
[8] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_15 =  int(sorted(list(data1[data1['Chip']  == die_8].Distance))[0])
dist_15_info = sorted(list(data1[data1['Chip']  == die_8].Distance))

Leadframe_8 = list(data1[data1['Chip']  == die_8][data1[data1['Chip']  == die_8].Distance == f_15].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_8].Distance))[0]
dist_15_info_m = sorted(list(data1[data1['Chip']  == die_8].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_15_info_m, count)
ran = map(int, ran)
for t_16 in ran:
    t_16 = t_16

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_15 = f_15
    else:
        dist_15 = t_16                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_8].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_16 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_8].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_8].Distance)))))[0]
dist_16_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_8].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_9 = list(data1[data1['Leadframe']  == Leadframe_8][data1[data1['Leadframe']  == Leadframe_8].Distance == f_16].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_16_info_m, count)
ran = map(int, ran)
for t_16 in ran:
    t_16 = t_16

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_16 = f_16
else:
    dist_16 = t_16

data1 = data1[data1.Leadframe != Leadframe_8]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,17):
    exec('total_dist = total_dist + dist_' + str(a))

print("[8]", "die_8 = ", die_8, "Leadframe_8 = ", Leadframe_8, "die_9 = ", die_9, "\t", "total_dist = ", total_dist)
"""
===============================================
[9] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_17 =  int(sorted(list(data1[data1['Chip']  == die_9].Distance))[0])
dist_15_info = sorted(list(data1[data1['Chip']  == die_9].Distance))

Leadframe_8 = list(data1[data1['Chip']  == die_9][data1[data1['Chip']  == die_9].Distance == f_17].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_9].Distance))[0]
dist_15_info_m = sorted(list(data1[data1['Chip']  == die_9].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_15_info_m, count)
ran = map(int, ran)
for t_18 in ran:
    t_18 = t_18

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_17 = f_17
    else:
        dist_17 = t_18                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_9].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_18 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_8].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_8].Distance)))))[0]
dist_18_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_8].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_10 = list(data1[data1['Leadframe']  == Leadframe_8][data1[data1['Leadframe']  == Leadframe_8].Distance == f_18].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_18_info_m, count)
ran = map(int, ran)
for t_18 in ran:
    t_18 = t_18

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_18 = f_18
else:
    dist_18 = t_18

data1 = data1[data1.Leadframe != Leadframe_8]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,19):
    exec('total_dist = total_dist + dist_' + str(a))

print("[9]", "die_9 = ", die_9, "Leadframe_8 = ", Leadframe_8, "die_10 = ", die_10, "\t", "total_dist = ", total_dist)
"""
===============================================
[10] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_19 =  int(sorted(list(data1[data1['Chip']  == die_10].Distance))[0])
dist_15_info = sorted(list(data1[data1['Chip']  == die_10].Distance))

Leadframe_10 = list(data1[data1['Chip']  == die_10][data1[data1['Chip']  == die_10].Distance == f_19].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_10].Distance))[0]
dist_15_info_m = sorted(list(data1[data1['Chip']  == die_10].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_15_info_m, count)
ran = map(int, ran)
for t_20 in ran:
    t_20 = t_20

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_19 = f_19
    else:
        dist_19 = t_20                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_10].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_20 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_10].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_10].Distance)))))[0]
dist_20_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_10].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_11 = list(data1[data1['Leadframe']  == Leadframe_10][data1[data1['Leadframe']  == Leadframe_10].Distance == f_20].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_20_info_m, count)
ran = map(int, ran)
for t_20 in ran:
    t_20 = t_20

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_20 = f_20
else:
    dist_20 = t_20

data1 = data1[data1.Leadframe != Leadframe_10]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,21):
    exec('total_dist = total_dist + dist_' + str(a))

print("[10]", "\t", "die_10 = ", die_10, "\t", "Leadframe_10 = ", Leadframe_10, "\t", "die_11 = ", die_11, "\t", "total_dist = ", total_dist)
"""
===============================================
[11] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_21 =  int(sorted(list(data1[data1['Chip']  == die_11].Distance))[0])
dist_21_info = sorted(list(data1[data1['Chip']  == die_11].Distance))

Leadframe_11 = list(data1[data1['Chip']  == die_11][data1[data1['Chip']  == die_11].Distance == f_21].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_11].Distance))[0]
dist_21_info_m = sorted(list(data1[data1['Chip']  == die_11].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_15_info_m, count)
ran = map(int, ran)
for t_21 in ran:
    t_21 = t_21

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_21 = f_21
    else:
        dist_21 = t_21                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_11].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_22 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_11].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_11].Distance)))))[0]
dist_22_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_11].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_12 = list(data1[data1['Leadframe']  == Leadframe_11][data1[data1['Leadframe']  == Leadframe_11].Distance == f_22].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_22_info_m, count)
ran = map(int, ran)
for t_22 in ran:
    t_22 = t_22

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_22 = f_22
else:
    dist_22 = t_22

data1 = data1[data1.Leadframe != Leadframe_11]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,23):
    exec('total_dist = total_dist + dist_' + str(a))

print("[11]", "\t", "die_11 = ", die_11, "\t", "Leadframe_11 = ", Leadframe_11, "\t", "die_12 = ", die_12, "\t", "total_dist = ", total_dist)
"""
===============================================
[12] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_23 =  int(sorted(list(data1[data1['Chip']  == die_12].Distance))[0])
dist_23_info = sorted(list(data1[data1['Chip']  == die_12].Distance))

Leadframe_12 = list(data1[data1['Chip']  == die_12][data1[data1['Chip']  == die_12].Distance == f_23].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_12].Distance))[0]
dist_23_info_m = sorted(list(data1[data1['Chip']  == die_12].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_15_info_m, count)
ran = map(int, ran)
for t_23 in ran:
    t_23 = t_23

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_23 = f_23
    else:
        dist_23 = t_23                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_12].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_24 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_12].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_12].Distance)))))[0]
dist_24_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_12].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_13 = list(data1[data1['Leadframe']  == Leadframe_12][data1[data1['Leadframe']  == Leadframe_12].Distance == f_24].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_24_info_m, count)
ran = map(int, ran)
for t_24 in ran:
    t_24 = t_24

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_24 = f_24
else:
    dist_24 = t_24

data1 = data1[data1.Leadframe != Leadframe_12]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,25):
    exec('total_dist = total_dist + dist_' + str(a))

print("[12]", "\t", "die_12 = ", die_12, "\t", "Leadframe_12 = ", Leadframe_12, "\t", "die_13 = ", die_13, "\t", "total_dist = ", total_dist)
"""
===============================================
[13] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_25 =  int(sorted(list(data1[data1['Chip']  == die_13].Distance))[0])
dist_25_info = sorted(list(data1[data1['Chip']  == die_13].Distance))

Leadframe_13 = list(data1[data1['Chip']  == die_13][data1[data1['Chip']  == die_13].Distance == f_25].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_13].Distance))[0]
dist_25_info_m = sorted(list(data1[data1['Chip']  == die_13].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_15_info_m, count)
ran = map(int, ran)
for t_25 in ran:
    t_25 = t_25

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_25 = f_25
    else:
        dist_25 = t_25                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_13].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_26 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_13].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_13].Distance)))))[0]
dist_26_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_13].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_14 = list(data1[data1['Leadframe']  == Leadframe_13][data1[data1['Leadframe']  == Leadframe_13].Distance == f_26].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_26_info_m, count)
ran = map(int, ran)
for t_26 in ran:
    t_26 = t_26

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_26 = f_26
else:
    dist_26 = t_26

data1 = data1[data1.Leadframe != Leadframe_13]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,27):
    exec('total_dist = total_dist + dist_' + str(a))

print("[13]", "\t", "die_13 = ", die_13, "\t", "Leadframe_13 = ", Leadframe_13, "\t", "die_14 = ", die_14, "\t", "total_dist = ", total_dist)
"""
===============================================
[14] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_27 =  int(sorted(list(data1[data1['Chip']  == die_14].Distance))[0])
dist_27_info = sorted(list(data1[data1['Chip']  == die_14].Distance))

Leadframe_14 = list(data1[data1['Chip']  == die_14][data1[data1['Chip']  == die_14].Distance == f_27].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_14].Distance))[0]
dist_27_info_m = sorted(list(data1[data1['Chip']  == die_14].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_15_info_m, count)
ran = map(int, ran)
for t_27 in ran:
    t_27 = t_27

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_27 = f_27
    else:
        dist_27 = t_27                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_14].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_28 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_14].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_14].Distance)))))[0]
dist_28_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_14].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_15 = list(data1[data1['Leadframe']  == Leadframe_14][data1[data1['Leadframe']  == Leadframe_14].Distance == f_28].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_28_info_m, count)
ran = map(int, ran)
for t_28 in ran:
    t_28 = t_28

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_28 = f_28
else:
    dist_28 = t_28

data1 = data1[data1.Leadframe != Leadframe_14]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,29):
    exec('total_dist = total_dist + dist_' + str(a))

print("[14]", "\t", "die_14 = ", die_14, "\t", "Leadframe_14 = ", Leadframe_14, "\t", "die_15 = ", die_15, "\t", "total_dist = ", total_dist)
"""
===============================================
[15] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_29 =  int(sorted(list(data1[data1['Chip']  == die_15].Distance))[0])
dist_29_info = sorted(list(data1[data1['Chip']  == die_15].Distance))

Leadframe_15 = list(data1[data1['Chip']  == die_15][data1[data1['Chip']  == die_15].Distance == f_29].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_15].Distance))[0]
dist_29_info_m = sorted(list(data1[data1['Chip']  == die_15].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_15_info_m, count)
ran = map(int, ran)
for t_29 in ran:
    t_29 = t_29

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_29 = f_29
    else:
        dist_29 = t_29                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_15].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_30 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_15].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_15].Distance)))))[0]
dist_30_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_15].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_16 = list(data1[data1['Leadframe']  == Leadframe_15][data1[data1['Leadframe']  == Leadframe_15].Distance == f_30].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_30_info_m, count)
ran = map(int, ran)
for t_30 in ran:
    t_30 = t_30

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_30 = f_30
else:
    dist_30 = t_30

data1 = data1[data1.Leadframe != Leadframe_15]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,31):
    exec('total_dist = total_dist + dist_' + str(a))

print("[15]", "\t", "die_15 = ", die_15, "\t", "Leadframe_15 = ", Leadframe_15, "\t", "die_16 = ", die_16, "\t", "total_dist = ", total_dist)
"""
===============================================
[16] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_31 =  int(sorted(list(data1[data1['Chip']  == die_16].Distance))[0])
dist_31_info = sorted(list(data1[data1['Chip']  == die_16].Distance))

Leadframe_16 = list(data1[data1['Chip']  == die_16][data1[data1['Chip']  == die_16].Distance == f_31].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_16].Distance))[0]
dist_31_info_m = sorted(list(data1[data1['Chip']  == die_16].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_15_info_m, count)
ran = map(int, ran)
for t_31 in ran:
    t_31 = t_31

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_31 = f_31
    else:
        dist_31 = t_31                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_16].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_32 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_16].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_16].Distance)))))[0]
dist_32_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_16].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_17 = list(data1[data1['Leadframe']  == Leadframe_16][data1[data1['Leadframe']  == Leadframe_16].Distance == f_32].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_32_info_m, count)
ran = map(int, ran)
for t_32 in ran:
    t_32 = t_32

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_32 = f_32
else:
    dist_32 = t_32

data1 = data1[data1.Leadframe != Leadframe_16]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,33):
    exec('total_dist = total_dist + dist_' + str(a))

print("[16]", "\t", "die_16 = ", die_16, "\t", "Leadframe_16 = ", Leadframe_16, "\t", "die_17 = ", die_17, "\t", "total_dist = ", total_dist)
"""
===============================================
[17] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_33 =  int(sorted(list(data1[data1['Chip']  == die_17].Distance))[0])
dist_33_info = sorted(list(data1[data1['Chip']  == die_17].Distance))

Leadframe_17 = list(data1[data1['Chip']  == die_17][data1[data1['Chip']  == die_17].Distance == f_33].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_17].Distance))[0]
dist_33_info_m = sorted(list(data1[data1['Chip']  == die_17].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_15_info_m, count)
ran = map(int, ran)
for t_33 in ran:
    t_33 = t_33

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_33 = f_33
    else:
        dist_33 = t_33                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_17].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_34 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_17].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_17].Distance)))))[0]
dist_34_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_17].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_18 = list(data1[data1['Leadframe']  == Leadframe_17][data1[data1['Leadframe']  == Leadframe_17].Distance == f_34].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_34_info_m, count)
ran = map(int, ran)
for t_34 in ran:
    t_34 = t_34

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_34 = f_34
else:
    dist_34 = t_34

data1 = data1[data1.Leadframe != Leadframe_17]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,35):
    exec('total_dist = total_dist + dist_' + str(a))

print("[17]", "\t", "die_17 = ", die_17, "\t", "Leadframe_17 = ", Leadframe_17, "\t", "die_18 = ", die_18, "\t", "total_dist = ", total_dist)
"""
===============================================
[18] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_35 =  int(sorted(list(data1[data1['Chip']  == die_18].Distance))[0])
dist_35_info = sorted(list(data1[data1['Chip']  == die_18].Distance))

Leadframe_18 = list(data1[data1['Chip']  == die_18][data1[data1['Chip']  == die_18].Distance == f_35].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_18].Distance))[0]
dist_35_info_m = sorted(list(data1[data1['Chip']  == die_18].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_15_info_m, count)
ran = map(int, ran)
for t_35 in ran:
    t_35 = t_35

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_35 = f_35
    else:
        dist_35 = t_35                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_18].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_36 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_18].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_18].Distance)))))[0]
dist_36_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_18].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_19 = list(data1[data1['Leadframe']  == Leadframe_18][data1[data1['Leadframe']  == Leadframe_18].Distance == f_36].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_36_info_m, count)
ran = map(int, ran)
for t_36 in ran:
    t_36 = t_36

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_36 = f_36
else:
    dist_36 = t_36

data1 = data1[data1.Leadframe != Leadframe_18]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,37):
    exec('total_dist = total_dist + dist_' + str(a))

print("[18]", "\t", "die_18 = ", die_18, "\t", "Leadframe_18 = ", Leadframe_18, "\t", "die_19 = ", die_19, "\t", "total_dist = ", total_dist)
"""
===============================================
[19] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_37 =  int(sorted(list(data1[data1['Chip']  == die_19].Distance))[0])
dist_37_info = sorted(list(data1[data1['Chip']  == die_19].Distance))

Leadframe_19 = list(data1[data1['Chip']  == die_19][data1[data1['Chip']  == die_19].Distance == f_37].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_19].Distance))[0]
dist_37_info_m = sorted(list(data1[data1['Chip']  == die_19].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_15_info_m, count)
ran = map(int, ran)
for t_37 in ran:
    t_37 = t_37

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_37 = f_37
    else:
        dist_37 = t_37                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_19].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_38 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_19].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_19].Distance)))))[0]
dist_38_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_19].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_20 = list(data1[data1['Leadframe']  == Leadframe_19][data1[data1['Leadframe']  == Leadframe_19].Distance == f_38].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_38_info_m, count)
ran = map(int, ran)
for t_38 in ran:
    t_38 = t_38

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_38 = f_38
else:
    dist_38 = t_38

data1 = data1[data1.Leadframe != Leadframe_19]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,39):
    exec('total_dist = total_dist + dist_' + str(a))

print("[19]", "\t", "die_19 = ", die_19, "\t", "Leadframe_19 = ", Leadframe_19, "\t", "die_20 = ", die_20, "\t", "total_dist = ", total_dist)

"""
===============================================
[20] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_39 =  int(sorted(list(data1[data1['Chip']  == die_20].Distance))[0])
dist_39_info = sorted(list(data1[data1['Chip']  == die_20].Distance))

Leadframe_20 = list(data1[data1['Chip']  == die_20][data1[data1['Chip']  == die_20].Distance == f_39].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_20].Distance))[0]
dist_39_info_m = sorted(list(data1[data1['Chip']  == die_20].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_15_info_m, count)
ran = map(int, ran)
for t_39 in ran:
    t_39 = t_39

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_39 = f_39
    else:
        dist_39 = t_39                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_20].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_40 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_20].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_20].Distance)))))[0]
dist_40_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_20].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_21 = list(data1[data1['Leadframe']  == Leadframe_20][data1[data1['Leadframe']  == Leadframe_20].Distance == f_40].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_40_info_m, count)
ran = map(int, ran)
for t_40 in ran:
    t_40 = t_40

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_40 = f_40
else:
    dist_40 = t_40

data1 = data1[data1.Leadframe != Leadframe_20]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,41):
    exec('total_dist = total_dist + dist_' + str(a))

print("[20]", "\t", "die_20 = ", die_20, "\t", "Leadframe_20 = ", Leadframe_20, "\t", "die_21 = ", die_21, "\t", "total_dist = ", total_dist)


"""
===============================================
[21] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_41 =  int(sorted(list(data1[data1['Chip']  == die_21].Distance))[0])
dist_41_info = sorted(list(data1[data1['Chip']  == die_21].Distance))

Leadframe_21 = list(data1[data1['Chip']  == die_21][data1[data1['Chip']  == die_21].Distance == f_41].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_21].Distance))[0]
dist_41_info_m = sorted(list(data1[data1['Chip']  == die_21].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_15_info_m, count)
ran = map(int, ran)
for t_41 in ran:
    t_41 = t_41

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_41 = f_41
    else:
        dist_41 = t_41                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_21].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_42 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_21].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_21].Distance)))))[0]
dist_42_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_21].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_22 = list(data1[data1['Leadframe']  == Leadframe_21][data1[data1['Leadframe']  == Leadframe_21].Distance == f_42].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_42_info_m, count)
ran = map(int, ran)
for t_42 in ran:
    t_42 = t_42

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_42 = f_42
else:
    dist_42 = t_42

data1 = data1[data1.Leadframe != Leadframe_21]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,43):
    exec('total_dist = total_dist + dist_' + str(a))

print("[21]", "\t", "die_21 = ", die_21, "\t", "Leadframe_21 = ", Leadframe_21, "\t", "die_22 = ", die_22, "\t", "total_dist = ", total_dist)
"""
===============================================
[22] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_43 =  int(sorted(list(data1[data1['Chip']  == die_22].Distance))[0])
dist_43_info = sorted(list(data1[data1['Chip']  == die_22].Distance))

Leadframe_22 = list(data1[data1['Chip']  == die_22][data1[data1['Chip']  == die_22].Distance == f_43].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_22].Distance))[0]
dist_43_info_m = sorted(list(data1[data1['Chip']  == die_22].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_15_info_m, count)
ran = map(int, ran)
for t_43 in ran:
    t_43 = t_43

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_43 = f_43
    else:
        dist_43 = t_43                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_22].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_44 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_22].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_22].Distance)))))[0]
dist_44_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_22].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_23 = list(data1[data1['Leadframe']  == Leadframe_22][data1[data1['Leadframe']  == Leadframe_22].Distance == f_44].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_44_info_m, count)
ran = map(int, ran)
for t_44 in ran:
    t_44 = t_44

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_44 = f_44
else:
    dist_44 = t_44

data1 = data1[data1.Leadframe != Leadframe_22]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,45):
    exec('total_dist = total_dist + dist_' + str(a))

print("[22]", "\t", "die_22 = ", die_22, "\t", "Leadframe_22 = ", Leadframe_22, "\t", "die_23 = ", die_23, "\t", "total_dist = ", total_dist)
"""
===============================================
[23] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_45 =  int(sorted(list(data1[data1['Chip']  == die_23].Distance))[0])
dist_45_info = sorted(list(data1[data1['Chip']  == die_23].Distance))

Leadframe_23 = list(data1[data1['Chip']  == die_23][data1[data1['Chip']  == die_23].Distance == f_45].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_23].Distance))[0]
dist_45_info_m = sorted(list(data1[data1['Chip']  == die_23].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_15_info_m, count)
ran = map(int, ran)
for t_45 in ran:
    t_45 = t_45

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_45 = f_45
    else:
        dist_45 = t_45                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_23].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_46 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_23].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_23].Distance)))))[0]
dist_46_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_23].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_24 = list(data1[data1['Leadframe']  == Leadframe_23][data1[data1['Leadframe']  == Leadframe_23].Distance == f_46].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_46_info_m, count)
ran = map(int, ran)
for t_46 in ran:
    t_46 = t_46

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_46 = f_46
else:
    dist_46 = t_46

data1 = data1[data1.Leadframe != Leadframe_23]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,47):
    exec('total_dist = total_dist + dist_' + str(a))

print("[23]", "\t", "die_23 = ", die_23, "\t", "Leadframe_23 = ", Leadframe_23, "\t", "die_24 = ", die_24, "\t", "total_dist = ", total_dist)
"""
===============================================
[24] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_47 =  int(sorted(list(data1[data1['Chip']  == die_24].Distance))[0])
dist_47_info = sorted(list(data1[data1['Chip']  == die_24].Distance))

Leadframe_24 = list(data1[data1['Chip']  == die_24][data1[data1['Chip']  == die_24].Distance == f_47].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_24].Distance))[0]
dist_47_info_m = sorted(list(data1[data1['Chip']  == die_24].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_15_info_m, count)
ran = map(int, ran)
for t_47 in ran:
    t_47 = t_47

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_47 = f_47
    else:
        dist_47 = t_47                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_24].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_48 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_24].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_24].Distance)))))[0]
dist_48_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_24].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_25 = list(data1[data1['Leadframe']  == Leadframe_24][data1[data1['Leadframe']  == Leadframe_24].Distance == f_48].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_48_info_m, count)
ran = map(int, ran)
for t_48 in ran:
    t_48 = t_48

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_48 = f_48
else:
    dist_48 = t_48

data1 = data1[data1.Leadframe != Leadframe_24]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,49):
    exec('total_dist = total_dist + dist_' + str(a))

print("[24]", "\t", "die_24 = ", die_24, "\t", "Leadframe_24 = ", Leadframe_24, "\t", "die_25 = ", die_25, "\t", "total_dist = ", total_dist)
"""
===============================================
[25] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_49 =  int(sorted(list(data1[data1['Chip']  == die_25].Distance))[0])
dist_49_info = sorted(list(data1[data1['Chip']  == die_25].Distance))

Leadframe_25 = list(data1[data1['Chip']  == die_25][data1[data1['Chip']  == die_25].Distance == f_49].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_25].Distance))[0]
dist_49_info_m = sorted(list(data1[data1['Chip']  == die_25].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_15_info_m, count)
ran = map(int, ran)
for t_49 in ran:
    t_49 = t_49

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_49 = f_49
    else:
        dist_49 = t_49                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_25].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_50 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_25].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_25].Distance)))))[0]
dist_50_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_25].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_26 = list(data1[data1['Leadframe']  == Leadframe_25][data1[data1['Leadframe']  == Leadframe_25].Distance == f_50].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_50_info_m, count)
ran = map(int, ran)
for t_50 in ran:
    t_50 = t_50

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_50 = f_50
else:
    dist_50 = t_50

data1 = data1[data1.Leadframe != Leadframe_25]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,51):
    exec('total_dist = total_dist + dist_' + str(a))

print("[25]", "\t", "die_25 = ", die_25, "\t", "Leadframe_25 = ", Leadframe_25, "\t", "die_26 = ", die_26, "\t", "total_dist = ", total_dist)
"""
===============================================
[26] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_51 =  int(sorted(list(data1[data1['Chip']  == die_26].Distance))[0])
dist_51_info = sorted(list(data1[data1['Chip']  == die_26].Distance))

Leadframe_26 = list(data1[data1['Chip']  == die_26][data1[data1['Chip']  == die_26].Distance == f_51].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_26].Distance))[0]
dist_51_info_m = sorted(list(data1[data1['Chip']  == die_26].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_15_info_m, count)
ran = map(int, ran)
for t_51 in ran:
    t_51 = t_51

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_51 = f_51
    else:
        dist_51 = t_51                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_26].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_52 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_26].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_26].Distance)))))[0]
dist_52_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_26].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_27 = list(data1[data1['Leadframe']  == Leadframe_26][data1[data1['Leadframe']  == Leadframe_26].Distance == f_52].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_52_info_m, count)
ran = map(int, ran)
for t_52 in ran:
    t_52 = t_52

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_52 = f_52
else:
    dist_52 = t_52

data1 = data1[data1.Leadframe != Leadframe_26]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,53):
    exec('total_dist = total_dist + dist_' + str(a))

print("[26]", "\t", "die_26 = ", die_26, "\t", "Leadframe_26 = ", Leadframe_26, "\t", "die_27 = ", die_27, "\t", "total_dist = ", total_dist)
"""
===============================================
[27] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_53 =  int(sorted(list(data1[data1['Chip']  == die_27].Distance))[0])
dist_53_info = sorted(list(data1[data1['Chip']  == die_27].Distance))

Leadframe_27 = list(data1[data1['Chip']  == die_27][data1[data1['Chip']  == die_27].Distance == f_53].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_27].Distance))[0]
dist_53_info_m = sorted(list(data1[data1['Chip']  == die_27].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_15_info_m, count)
ran = map(int, ran)
for t_53 in ran:
    t_53 = t_53

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_53 = f_53
    else:
        dist_53 = t_53                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_27].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_54 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_27].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_27].Distance)))))[0]
dist_54_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_27].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_28 = list(data1[data1['Leadframe']  == Leadframe_27][data1[data1['Leadframe']  == Leadframe_27].Distance == f_54].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_54_info_m, count)
ran = map(int, ran)
for t_54 in ran:
    t_54 = t_54

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_54 = f_54
else:
    dist_54 = t_54

data1 = data1[data1.Leadframe != Leadframe_27]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,55):
    exec('total_dist = total_dist + dist_' + str(a))

print("[27]", "\t", "die_27 = ", die_27, "\t", "Leadframe_27 = ", Leadframe_27, "\t", "die_28 = ", die_28, "\t", "total_dist = ", total_dist)

"""
===============================================
[28] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_55 =  int(sorted(list(data1[data1['Chip']  == die_28].Distance))[0])
dist_55_info = sorted(list(data1[data1['Chip']  == die_28].Distance))

Leadframe_28 = list(data1[data1['Chip']  == die_28][data1[data1['Chip']  == die_28].Distance == f_55].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_28].Distance))[0]
dist_55_info_m = sorted(list(data1[data1['Chip']  == die_28].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_15_info_m, count)
ran = map(int, ran)
for t_55 in ran:
    t_55 = t_55

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_55 = f_55
    else:
        dist_55 = t_55                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_28].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_56 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_28].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_28].Distance)))))[0]
dist_56_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_28].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_29 = list(data1[data1['Leadframe']  == Leadframe_28][data1[data1['Leadframe']  == Leadframe_28].Distance == f_56].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_56_info_m, count)
ran = map(int, ran)
for t_56 in ran:
    t_56 = t_56

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_56 = f_56
else:
    dist_56 = t_56

data1 = data1[data1.Leadframe != Leadframe_28]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,57):
    exec('total_dist = total_dist + dist_' + str(a))

print("[28]", "\t", "die_28 = ", die_28, "\t", "Leadframe_28 = ", Leadframe_28, "\t", "die_29 = ", die_29, "\t", "total_dist = ", total_dist)
"""
===============================================
[29] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_57 =  int(sorted(list(data1[data1['Chip']  == die_29].Distance))[0])
dist_57_info = sorted(list(data1[data1['Chip']  == die_29].Distance))

Leadframe_29 = list(data1[data1['Chip']  == die_29][data1[data1['Chip']  == die_29].Distance == f_57].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_29].Distance))[0]
dist_57_info_m = sorted(list(data1[data1['Chip']  == die_29].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_15_info_m, count)
ran = map(int, ran)
for t_57 in ran:
    t_57 = t_57

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_57 = f_57
    else:
        dist_57 = t_57                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_29].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_58 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_29].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_29].Distance)))))[0]
dist_58_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_29].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_30 = list(data1[data1['Leadframe']  == Leadframe_29][data1[data1['Leadframe']  == Leadframe_29].Distance == f_58].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_58_info_m, count)
ran = map(int, ran)
for t_58 in ran:
    t_58 = t_58

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_58 = f_58
else:
    dist_58 = t_58

data1 = data1[data1.Leadframe != Leadframe_29]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,59):
    exec('total_dist = total_dist + dist_' + str(a))
print("[29]", "\t", "die_29 = ", die_29, "\t", "Leadframe_29 = ", Leadframe_29, "\t", "die_30 = ", die_30, "\t", "total_dist = ", total_dist)
"""
===============================================
[30] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_59 =  int(sorted(list(data1[data1['Chip']  == die_30].Distance))[0])
dist_59_info = sorted(list(data1[data1['Chip']  == die_30].Distance))

Leadframe_30 = list(data1[data1['Chip']  == die_30][data1[data1['Chip']  == die_30].Distance == f_59].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_30].Distance))[0]
dist_59_info_m = sorted(list(data1[data1['Chip']  == die_30].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_15_info_m, count)
ran = map(int, ran)
for t_59 in ran:
    t_59 = t_59

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_59 = f_59
    else:
        dist_59 = t_59                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_30].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_60 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_30].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_30].Distance)))))[0]
dist_60_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_30].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_31 = list(data1[data1['Leadframe']  == Leadframe_30][data1[data1['Leadframe']  == Leadframe_30].Distance == f_60].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_60_info_m, count)
ran = map(int, ran)
for t_60 in ran:
    t_60 = t_60

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_60 = f_60
else:
    dist_60 = t_60

data1 = data1[data1.Leadframe != Leadframe_30]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,61):
    exec('total_dist = total_dist + dist_' + str(a))
print("[30]", "\t", "die_30 = ", die_30, "\t", "Leadframe_30 = ", Leadframe_30, "\t", "die_31 = ", die_31, "\t", "total_dist = ", total_dist)
"""
===============================================
[31] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_61 =  int(sorted(list(data1[data1['Chip']  == die_31].Distance))[0])
dist_61_info = sorted(list(data1[data1['Chip']  == die_31].Distance))

Leadframe_31 = list(data1[data1['Chip']  == die_31][data1[data1['Chip']  == die_31].Distance == f_61].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_31].Distance))[0]
dist_61_info_m = sorted(list(data1[data1['Chip']  == die_31].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_15_info_m, count)
ran = map(int, ran)
for t_61 in ran:
    t_61 = t_61

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_61 = f_61
    else:
        dist_61 = t_61                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_31].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_62 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_31].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_31].Distance)))))[0]
dist_62_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_31].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_32 = list(data1[data1['Leadframe']  == Leadframe_31][data1[data1['Leadframe']  == Leadframe_31].Distance == f_62].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_62_info_m, count)
ran = map(int, ran)
for t_62 in ran:
    t_62 = t_62

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_62 = f_62
else:
    dist_62 = t_62

data1 = data1[data1.Leadframe != Leadframe_31]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,63):
    exec('total_dist = total_dist + dist_' + str(a))
print("[31]", "\t", "die_31 = ", die_31, "\t", "Leadframe_31 = ", Leadframe_31, "\t", "die_32 = ", die_32, "\t", "total_dist = ", total_dist)
"""
===============================================
[32] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_63 =  int(sorted(list(data1[data1['Chip']  == die_32].Distance))[0])
dist_63_info = sorted(list(data1[data1['Chip']  == die_32].Distance))

Leadframe_32 = list(data1[data1['Chip']  == die_32][data1[data1['Chip']  == die_32].Distance == f_63].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_32].Distance))[0]
dist_63_info_m = sorted(list(data1[data1['Chip']  == die_32].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_63_info_m, count)
ran = map(int, ran)
for t_63 in ran:
    t_63 = t_63

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_63 = f_63
    else:
        dist_63 = t_63                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_32].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_64 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_32].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_32].Distance)))))[0]
dist_64_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_32].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_33 = list(data1[data1['Leadframe']  == Leadframe_32][data1[data1['Leadframe']  == Leadframe_32].Distance == f_64].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_64_info_m, count)
ran = map(int, ran)
for t_64 in ran:
    t_64 = t_64

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_64 = f_64
else:
    dist_64 = t_64

data1 = data1[data1.Leadframe != Leadframe_32]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,65):
    exec('total_dist = total_dist + dist_' + str(a))
print("[32]", "\t", "die_32 = ", die_32, "\t", "Leadframe_32 = ", Leadframe_32, "\t", "die_33 = ", die_33, "\t", "total_dist = ", total_dist)
"""
===============================================
[33] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_65 =  int(sorted(list(data1[data1['Chip']  == die_33].Distance))[0])
dist_65_info = sorted(list(data1[data1['Chip']  == die_33].Distance))

Leadframe_33 = list(data1[data1['Chip']  == die_33][data1[data1['Chip']  == die_33].Distance == f_65].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_33].Distance))[0]
dist_65_info_m = sorted(list(data1[data1['Chip']  == die_33].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_65_info_m, count)
ran = map(int, ran)
for t_65 in ran:
    t_65 = t_65

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_65 = f_65
    else:
        dist_65 = t_65                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_33].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_66 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_33].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_33].Distance)))))[0]
dist_66_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_33].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_34 = list(data1[data1['Leadframe']  == Leadframe_33][data1[data1['Leadframe']  == Leadframe_33].Distance == f_66].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_66_info_m, count)
ran = map(int, ran)
for t_66 in ran:
    t_66 = t_66

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_66 = f_66
else:
    dist_66 = t_66

data1 = data1[data1.Leadframe != Leadframe_33]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,67):
    exec('total_dist = total_dist + dist_' + str(a))
print("[33]", "\t", "die_33 = ", die_33, "\t", "Leadframe_33 = ", Leadframe_33, "\t", "die_34 = ", die_34, "\t", "total_dist = ", total_dist)
"""
===============================================
[34] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_67 =  int(sorted(list(data1[data1['Chip']  == die_34].Distance))[0])
dist_67_info = sorted(list(data1[data1['Chip']  == die_34].Distance))

Leadframe_34 = list(data1[data1['Chip']  == die_34][data1[data1['Chip']  == die_34].Distance == f_67].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_34].Distance))[0]
dist_67_info_m = sorted(list(data1[data1['Chip']  == die_34].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_67_info_m, count)
ran = map(int, ran)
for t_67 in ran:
    t_67 = t_67

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_67 = f_67
    else:
        dist_67 = t_67                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_34].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_68 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_34].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_34].Distance)))))[0]
dist_68_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_34].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_35 = list(data1[data1['Leadframe']  == Leadframe_34][data1[data1['Leadframe']  == Leadframe_34].Distance == f_68].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_68_info_m, count)
ran = map(int, ran)
for t_68 in ran:
    t_68 = t_68

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_68 = f_68
else:
    dist_68 = t_68

data1 = data1[data1.Leadframe != Leadframe_34]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,69):
    exec('total_dist = total_dist + dist_' + str(a))
print("[34]", "\t", "die_34 = ", die_34, "\t", "Leadframe_34 = ", Leadframe_34, "\t", "die_35 = ", die_35, "\t", "total_dist = ", total_dist)
"""
===============================================
[35] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_69 =  int(sorted(list(data1[data1['Chip']  == die_35].Distance))[0])
dist_69_info = sorted(list(data1[data1['Chip']  == die_35].Distance))

Leadframe_35 = list(data1[data1['Chip']  == die_35][data1[data1['Chip']  == die_35].Distance == f_69].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_35].Distance))[0]
dist_69_info_m = sorted(list(data1[data1['Chip']  == die_35].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_69_info_m, count)
ran = map(int, ran)
for t_69 in ran:
    t_69 = t_69

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_69 = f_69
    else:
        dist_69 = t_69                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_35].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_70 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_35].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_35].Distance)))))[0]
dist_70_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_35].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_36 = list(data1[data1['Leadframe']  == Leadframe_35][data1[data1['Leadframe']  == Leadframe_35].Distance == f_70].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_70_info_m, count)
ran = map(int, ran)
for t_70 in ran:
    t_70 = t_70

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_70 = f_70
else:
    dist_70 = t_70

data1 = data1[data1.Leadframe != Leadframe_35]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,71):
    exec('total_dist = total_dist + dist_' + str(a))
print("[35]", "\t", "die_35 = ", die_35, "\t", "Leadframe_35 = ", Leadframe_35, "\t", "die_36 = ", die_36, "\t", "total_dist = ", total_dist)
"""
===============================================
[36] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_71 =  int(sorted(list(data1[data1['Chip']  == die_36].Distance))[0])
dist_71_info = sorted(list(data1[data1['Chip']  == die_36].Distance))

Leadframe_36 = list(data1[data1['Chip']  == die_36][data1[data1['Chip']  == die_36].Distance == f_71].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_36].Distance))[0]
dist_71_info_m = sorted(list(data1[data1['Chip']  == die_36].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_71_info_m, count)
ran = map(int, ran)
for t_71 in ran:
    t_71 = t_71

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_71 = f_71
    else:
        dist_71 = t_71                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_36].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_72 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_36].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_36].Distance)))))[0]
dist_72_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_36].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_37 = list(data1[data1['Leadframe']  == Leadframe_36][data1[data1['Leadframe']  == Leadframe_36].Distance == f_72].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_72_info_m, count)
ran = map(int, ran)
for t_72 in ran:
    t_72 = t_72

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_72 = f_72
else:
    dist_72 = t_72

data1 = data1[data1.Leadframe != Leadframe_36]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,73):
    exec('total_dist = total_dist + dist_' + str(a))
print("[36]", "\t", "die_36 = ", die_36, "\t", "Leadframe_36 = ", Leadframe_36, "\t", "die_37 = ", die_37, "\t", "total_dist = ", total_dist)
"""
===============================================
[37] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_73 =  int(sorted(list(data1[data1['Chip']  == die_37].Distance))[0])
dist_73_info = sorted(list(data1[data1['Chip']  == die_37].Distance))

Leadframe_37 = list(data1[data1['Chip']  == die_37][data1[data1['Chip']  == die_37].Distance == f_73].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_37].Distance))[0]
dist_73_info_m = sorted(list(data1[data1['Chip']  == die_37].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_73_info_m, count)
ran = map(int, ran)
for t_73 in ran:
    t_73 = t_73

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_73 = f_73
    else:
        dist_73 = t_73                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_37].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_74 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_37].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_37].Distance)))))[0]
dist_74_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_37].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_38 = list(data1[data1['Leadframe']  == Leadframe_37][data1[data1['Leadframe']  == Leadframe_37].Distance == f_74].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_74_info_m, count)
ran = map(int, ran)
for t_74 in ran:
    t_74 = t_74

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_74 = f_74
else:
    dist_74 = t_74

data1 = data1[data1.Leadframe != Leadframe_37]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,75):
    exec('total_dist = total_dist + dist_' + str(a))
print("[37]", "\t", "die_37 = ", die_37, "\t", "Leadframe_37 = ", Leadframe_37, "\t", "die_38 = ", die_38, "\t", "total_dist = ", total_dist)
"""
===============================================
[38] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_75 =  int(sorted(list(data1[data1['Chip']  == die_38].Distance))[0])
dist_75_info = sorted(list(data1[data1['Chip']  == die_38].Distance))

Leadframe_38 = list(data1[data1['Chip']  == die_38][data1[data1['Chip']  == die_38].Distance == f_75].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_38].Distance))[0]
dist_75_info_m = sorted(list(data1[data1['Chip']  == die_38].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_75_info_m, count)
ran = map(int, ran)
for t_75 in ran:
    t_75 = t_75

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_75 = f_75
    else:
        dist_75 = t_75                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_38].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_76 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_38].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_38].Distance)))))[0]
dist_76_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_38].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_39 = list(data1[data1['Leadframe']  == Leadframe_38][data1[data1['Leadframe']  == Leadframe_38].Distance == f_76].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_76_info_m, count)
ran = map(int, ran)
for t_76 in ran:
    t_76 = t_76

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_76 = f_76
else:
    dist_76 = t_76

data1 = data1[data1.Leadframe != Leadframe_38]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,77):
    exec('total_dist = total_dist + dist_' + str(a))
print("[38]", "\t", "die_38 = ", die_38, "\t", "Leadframe_38 = ", Leadframe_38, "\t", "die_39 = ", die_39, "\t", "total_dist = ", total_dist)
"""
===============================================
[39] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_77 =  int(sorted(list(data1[data1['Chip']  == die_39].Distance))[0])
dist_77_info = sorted(list(data1[data1['Chip']  == die_39].Distance))

Leadframe_39 = list(data1[data1['Chip']  == die_39][data1[data1['Chip']  == die_39].Distance == f_77].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_39].Distance))[0]
dist_77_info_m = sorted(list(data1[data1['Chip']  == die_39].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_77_info_m, count)
ran = map(int, ran)
for t_77 in ran:
    t_77 = t_77

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_77 = f_77
    else:
        dist_77 = t_77                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_39].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_78 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_39].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_39].Distance)))))[0]
dist_78_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_39].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_40 = list(data1[data1['Leadframe']  == Leadframe_39][data1[data1['Leadframe']  == Leadframe_39].Distance == f_78].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_78_info_m, count)
ran = map(int, ran)
for t_78 in ran:
    t_78 = t_78

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_78 = f_78
else:
    dist_78 = t_78

data1 = data1[data1.Leadframe != Leadframe_39]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,79):
    exec('total_dist = total_dist + dist_' + str(a))
print("[39]", "\t", "die_39 = ", die_39, "\t", "Leadframe_39 = ", Leadframe_39, "\t", "die_40 = ", die_40, "\t", "total_dist = ", total_dist)
"""
===============================================
[40] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""
f_79 =  int(sorted(list(data1[data1['Chip']  == die_40].Distance))[0])
dist_79_info = sorted(list(data1[data1['Chip']  == die_40].Distance))

Leadframe_40 = list(data1[data1['Chip']  == die_40][data1[data1['Chip']  == die_40].Distance == f_79].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data1[data1['Chip']  == die_40].Distance))[0]
dist_79_info_m = sorted(list(data1[data1['Chip']  == die_40].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_79_info_m, count)
ran = map(int, ran)
for t_79 in ran:
    t_79 = t_79

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_79 = f_79
    else:
        dist_79 = t_79                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data1 = data1.drop(data1[data1.Chip == die_40].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_80 = int( sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_40].Distance)))))[0])

del sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_40].Distance)))))[0]
dist_80_info_m = list(set(sorted(list(set(sorted(list(data1[data1['Leadframe']  == Leadframe_40].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_41 = list(data1[data1['Leadframe']  == Leadframe_40][data1[data1['Leadframe']  == Leadframe_40].Distance == f_80].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택
#print(die_41)

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_80_info_m, count)
ran = map(int, ran)
for t_80 in ran:
    t_80 = t_80

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_80 = f_80
else:
    dist_80 = t_80

data1 = data1[data1.Leadframe != Leadframe_40]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,81):
    exec('total_dist = total_dist + dist_' + str(a))
print("[40]", "\t", "die_40 = ", die_40, "\t", "Leadframe_40 = ", Leadframe_40, "\t", "die_41 = ", die_41, "\t", "total_dist = ", total_dist)
print("=======================================================================================")
"""
===============================================
[41] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_41_info = data2[data2.Chip  == die_41]

f_81 =  int(sorted(list(data2[data2['Chip']  == die_41].Distance))[0])
dist_81_info = sorted(list(data2[data2['Chip']  == die_41].Distance))

Leadframe_41 =  list(die_41_info[die_41_info.Distance == f_81].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_41 = list(data2[data2['Chip']  == die_41][data2[data2['Chip']  == die_41].Distance == f_81].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_41].Distance))[0]
dist_81_info_m = sorted(list(data2[data2['Chip']  == die_41].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_81_info_m, count)
ran = map(int, ran)
for t_81 in ran:
    t_81 = t_81

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_81 = f_81
    else:
        dist_81 = t_81                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_41].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_82 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_41].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_41].Distance)))))[0]
dist_82_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_41].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_42 = list(data2[data2['Leadframe']  == Leadframe_41][data2[data2['Leadframe']  == Leadframe_41].Distance == f_82].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_82_info_m, count)
ran = map(int, ran)
for t_82 in ran:
    t_82 = t_82

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_82 = f_82
else:
    dist_82 = t_82

data2 = data2[data2.Leadframe != Leadframe_41]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,83):
    exec('total_dist = total_dist + dist_' + str(a))
print("[41]", "\t", "die_41 = ", die_41, "\t", "Leadframe_41 = ", Leadframe_41, "\t", "die_42 = ", die_42, "\t", "total_dist = ", total_dist)
"""
===============================================
[42] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_42_info = data2[data2.Chip  == die_42]

f_83 =  int(sorted(list(data2[data2['Chip']  == die_42].Distance))[0])
dist_83_info = sorted(list(data2[data2['Chip']  == die_42].Distance))

Leadframe_42 =  list(die_42_info[die_42_info.Distance == f_83].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_42 = list(data2[data2['Chip']  == die_42][data2[data2['Chip']  == die_42].Distance == f_83].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_42].Distance))[0]
dist_83_info_m = sorted(list(data2[data2['Chip']  == die_42].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_83_info_m, count)
ran = map(int, ran)
for t_83 in ran:
    t_83 = t_83

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_83 = f_83
    else:
        dist_83 = t_83                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_42].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_84 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_42].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_42].Distance)))))[0]
dist_84_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_42].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_43 = list(data2[data2['Leadframe']  == Leadframe_42][data2[data2['Leadframe']  == Leadframe_42].Distance == f_84].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_84_info_m, count)
ran = map(int, ran)
for t_84 in ran:
    t_84 = t_84

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_84 = f_84
else:
    dist_84 = t_84

data2 = data2[data2.Leadframe != Leadframe_42]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제

total_dist = 0
for a in range(1,85):
    exec('total_dist = total_dist + dist_' + str(a))
print("[42]", "\t", "die_42 = ", die_42, "\t", "Leadframe_42 = ", Leadframe_42, "\t", "die_43 = ", die_43, "\t", "total_dist = ", total_dist)
"""
===============================================
[43] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_43_info = data2[data2.Chip  == die_43]

f_85 =  int(sorted(list(data2[data2['Chip']  == die_43].Distance))[0])
dist_85_info = sorted(list(data2[data2['Chip']  == die_43].Distance))

Leadframe_43 =  list(die_43_info[die_43_info.Distance == f_85].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_43 = list(data2[data2['Chip']  == die_43][data2[data2['Chip']  == die_43].Distance == f_85].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_43].Distance))[0]
dist_85_info_m = sorted(list(data2[data2['Chip']  == die_43].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_85_info_m, count)
ran = map(int, ran)
for t_85 in ran:
    t_85 = t_85

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_85 = f_85
    else:
        dist_85 = t_85                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_43].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_86 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_43].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_43].Distance)))))[0]
dist_86_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_43].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_44 = list(data2[data2['Leadframe']  == Leadframe_43][data2[data2['Leadframe']  == Leadframe_43].Distance == f_86].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_86_info_m, count)
ran = map(int, ran)
for t_86 in ran:
    t_86 = t_86

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_86 = f_86
else:
    dist_86 = t_86

data2 = data2[data2.Leadframe != Leadframe_43]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = 0
for a in range(1,87):
    exec('total_dist = total_dist + dist_' + str(a))
print("[43]", "\t", "die_43 = ", die_43, "\t", "Leadframe_43 = ", Leadframe_43, "\t", "die_44 = ", die_44, "\t", "total_dist = ", total_dist)
"""
===============================================
[44] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_44_info = data2[data2.Chip  == die_44]

f_87 =  int(sorted(list(data2[data2['Chip']  == die_44].Distance))[0])
dist_87_info = sorted(list(data2[data2['Chip']  == die_44].Distance))

Leadframe_44 =  list(die_44_info[die_44_info.Distance == f_87].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_44 = list(data2[data2['Chip']  == die_44][data2[data2['Chip']  == die_44].Distance == f_87].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_44].Distance))[0]
dist_87_info_m = sorted(list(data2[data2['Chip']  == die_44].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_87_info_m, count)
ran = map(int, ran)
for t_87 in ran:
    t_87 = t_87

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_87 = f_87
    else:
        dist_87 = t_87                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_44].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_88 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_44].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_44].Distance)))))[0]
dist_88_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_44].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_45 = list(data2[data2['Leadframe']  == Leadframe_44][data2[data2['Leadframe']  == Leadframe_44].Distance == f_88].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_88_info_m, count)
ran = map(int, ran)
for t_88 in ran:
    t_88 = t_88

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_88 = f_88
else:
    dist_88 = t_88

data2 = data2[data2.Leadframe != Leadframe_44]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = 0
for a in range(1,89):
    exec('total_dist = total_dist + dist_' + str(a))
print("[44]", "\t", "die_44 = ", die_44, "\t", "Leadframe_44 = ", Leadframe_44, "\t", "die_45 = ", die_45, "\t", "total_dist = ", total_dist)
"""
===============================================
[45] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_45_info = data2[data2.Chip  == die_45]

f_89 =  int(sorted(list(data2[data2['Chip']  == die_45].Distance))[0])
dist_89_info = sorted(list(data2[data2['Chip']  == die_45].Distance))

Leadframe_45 =  list(die_45_info[die_45_info.Distance == f_89].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_45 = list(data2[data2['Chip']  == die_45][data2[data2['Chip']  == die_45].Distance == f_89].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_45].Distance))[0]
dist_89_info_m = sorted(list(data2[data2['Chip']  == die_45].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_89_info_m, count)
ran = map(int, ran)
for t_89 in ran:
    t_89 = t_89

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_89 = f_89
    else:
        dist_89 = t_89                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_45].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_90 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_45].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_45].Distance)))))[0]
dist_90_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_45].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_46 = list(data2[data2['Leadframe']  == Leadframe_45][data2[data2['Leadframe']  == Leadframe_45].Distance == f_90].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_90_info_m, count)
ran = map(int, ran)
for t_90 in ran:
    t_90 = t_90

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_90 = f_90
else:
    dist_90 = t_90

data2 = data2[data2.Leadframe != Leadframe_45]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = 0
for a in range(1,91):
    exec('total_dist = total_dist + dist_' + str(a))
print("[45]", "\t", "die_45 = ", die_45, "\t", "Leadframe_45 = ", Leadframe_45, "\t", "die_46 = ", die_46, "\t", "total_dist = ", total_dist)
"""
===============================================
[46] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_46_info = data2[data2.Chip  == die_46]

f_91 =  int(sorted(list(data2[data2['Chip']  == die_46].Distance))[0])
dist_91_info = sorted(list(data2[data2['Chip']  == die_46].Distance))

Leadframe_46 =  list(die_46_info[die_46_info.Distance == f_91].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_46 = list(data2[data2['Chip']  == die_46][data2[data2['Chip']  == die_46].Distance == f_91].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_46].Distance))[0]
dist_91_info_m = sorted(list(data2[data2['Chip']  == die_46].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_91_info_m, count)
ran = map(int, ran)
for t_91 in ran:
    t_91 = t_91

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_91 = f_91
    else:
        dist_91 = t_91                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_46].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_92 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_46].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_46].Distance)))))[0]
dist_92_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_46].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_47 = list(data2[data2['Leadframe']  == Leadframe_46][data2[data2['Leadframe']  == Leadframe_46].Distance == f_92].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_92_info_m, count)
ran = map(int, ran)
for t_92 in ran:
    t_92 = t_92

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_92 = f_92
else:
    dist_92 = t_92

data2 = data2[data2.Leadframe != Leadframe_46]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = 0
for a in range(1,93):
    exec('total_dist = total_dist + dist_' + str(a))
print("[46]", "\t", "die_46 = ", die_46, "\t", "Leadframe_46 = ", Leadframe_46, "\t", "die_47 = ", die_47, "\t", "total_dist = ", total_dist)
"""
===============================================
[47] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_47_info = data2[data2.Chip  == die_47]

f_93 =  int(sorted(list(data2[data2['Chip']  == die_47].Distance))[0])
dist_93_info = sorted(list(data2[data2['Chip']  == die_47].Distance))

Leadframe_47 =  list(die_47_info[die_47_info.Distance == f_93].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_47 = list(data2[data2['Chip']  == die_47][data2[data2['Chip']  == die_47].Distance == f_93].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_47].Distance))[0]
dist_93_info_m = sorted(list(data2[data2['Chip']  == die_47].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_93_info_m, count)
ran = map(int, ran)
for t_93 in ran:
    t_93 = t_93

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_93 = f_93
    else:
        dist_93 = t_93                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_47].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_94 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_47].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_47].Distance)))))[0]
dist_94_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_47].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_48 = list(data2[data2['Leadframe']  == Leadframe_47][data2[data2['Leadframe']  == Leadframe_47].Distance == f_94].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_94_info_m, count)
ran = map(int, ran)
for t_94 in ran:
    t_94 = t_94

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_94 = f_94
else:
    dist_94 = t_94

data2 = data2[data2.Leadframe != Leadframe_47]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = 0
for a in range(1,95):
    exec('total_dist = total_dist + dist_' + str(a))
print("[47]", "\t", "die_47 = ", die_47, "\t", "Leadframe_47 = ", Leadframe_47, "\t", "die_48 = ", die_48, "\t", "total_dist = ", total_dist)
"""
===============================================
[48] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_48_info = data2[data2.Chip  == die_48]

f_95 =  int(sorted(list(data2[data2['Chip']  == die_48].Distance))[0])
dist_95_info = sorted(list(data2[data2['Chip']  == die_48].Distance))

Leadframe_48 =  list(die_48_info[die_48_info.Distance == f_95].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_48 = list(data2[data2['Chip']  == die_48][data2[data2['Chip']  == die_48].Distance == f_95].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_48].Distance))[0]
dist_95_info_m = sorted(list(data2[data2['Chip']  == die_48].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_95_info_m, count)
ran = map(int, ran)
for t_95 in ran:
    t_95 = t_95

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_95 = f_95
    else:
        dist_95 = t_95                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_48].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_96 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_48].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_48].Distance)))))[0]
dist_96_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_48].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_49 = list(data2[data2['Leadframe']  == Leadframe_48][data2[data2['Leadframe']  == Leadframe_48].Distance == f_96].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_96_info_m, count)
ran = map(int, ran)
for t_96 in ran:
    t_96 = t_96

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_96 = f_96
else:
    dist_96 = t_96

data2 = data2[data2.Leadframe != Leadframe_48]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = 0
for a in range(1,97):
    exec('total_dist = total_dist + dist_' + str(a))
print("[48]", "\t", "die_48 = ", die_48, "\t", "Leadframe_48 = ", Leadframe_48, "\t", "die_49 = ", die_49, "\t", "total_dist = ", total_dist)
"""
===============================================
[49] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_49_info = data2[data2.Chip  == die_49]

f_97 =  int(sorted(list(data2[data2['Chip']  == die_49].Distance))[0])
dist_97_info = sorted(list(data2[data2['Chip']  == die_49].Distance))

Leadframe_49 =  list(die_49_info[die_49_info.Distance == f_97].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_49 = list(data2[data2['Chip']  == die_49][data2[data2['Chip']  == die_49].Distance == f_97].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_49].Distance))[0]
dist_97_info_m = sorted(list(data2[data2['Chip']  == die_49].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_97_info_m, count)
ran = map(int, ran)
for t_97 in ran:
    t_97 = t_97

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_97 = f_97
    else:
        dist_97 = t_97                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_49].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_98 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_49].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_49].Distance)))))[0]
dist_98_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_49].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_50 = list(data2[data2['Leadframe']  == Leadframe_49][data2[data2['Leadframe']  == Leadframe_49].Distance == f_98].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_98_info_m, count)
ran = map(int, ran)
for t_98 in ran:
    t_98 = t_98

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_98 = f_98
else:
    dist_98 = t_98

data2 = data2[data2.Leadframe != Leadframe_49]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = 0
for a in range(1,99):
    exec('total_dist = total_dist + dist_' + str(a))
print("[49]", "\t", "die_49 = ", die_49, "\t", "Leadframe_49 = ", Leadframe_49, "\t", "die_50 = ", die_50, "\t", "total_dist = ", total_dist)
"""
===============================================
[50] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_50_info = data2[data2.Chip  == die_50]

f_99 =  int(sorted(list(data2[data2['Chip']  == die_50].Distance))[0])
dist_99_info = sorted(list(data2[data2['Chip']  == die_50].Distance))

Leadframe_50 =  list(die_50_info[die_50_info.Distance == f_99].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_50 = list(data2[data2['Chip']  == die_50][data2[data2['Chip']  == die_50].Distance == f_99].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_50].Distance))[0]
dist_99_info_m = sorted(list(data2[data2['Chip']  == die_50].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_99_info_m, count)
ran = map(int, ran)
for t_99 in ran:
    t_99 = t_99

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_99 = f_99
    else:
        dist_99 = t_99                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_50].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_100 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_50].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_50].Distance)))))[0]
dist_100_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_50].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_51 = list(data2[data2['Leadframe']  == Leadframe_50][data2[data2['Leadframe']  == Leadframe_50].Distance == f_100].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_100_info_m, count)
ran = map(int, ran)
for t_100 in ran:
    t_100 = t_100

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_100 = f_100
else:
    dist_100 = t_100

data2 = data2[data2.Leadframe != Leadframe_50]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = 0
for a in range(1,101):
    exec('total_dist = total_dist + dist_' + str(a))
print("[50]", "\t", "die_50 = ", die_50, "\t", "Leadframe_50 = ", Leadframe_50, "\t", "die_51 = ", die_51, "\t", "total_dist = ", total_dist)
"""
===============================================
[51] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_51_info = data2[data2.Chip  == die_51]

f_101 =  int(sorted(list(data2[data2['Chip']  == die_51].Distance))[0])
dist_101_info = sorted(list(data2[data2['Chip']  == die_51].Distance))

Leadframe_51 =  list(die_51_info[die_51_info.Distance == f_101].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_51 = list(data2[data2['Chip']  == die_51][data2[data2['Chip']  == die_51].Distance == f_101].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_51].Distance))[0]
dist_101_info_m = sorted(list(data2[data2['Chip']  == die_51].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_101_info_m, count)
ran = map(int, ran)
for t_101 in ran:
    t_101 = t_101

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_101 = f_101
    else:
        dist_101 = t_101                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_51].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_102 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_51].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_51].Distance)))))[0]
dist_102_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_51].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_52 = list(data2[data2['Leadframe']  == Leadframe_51][data2[data2['Leadframe']  == Leadframe_51].Distance == f_102].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_102_info_m, count)
ran = map(int, ran)
for t_102 in ran:
    t_102 = t_102

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_102 = f_102
else:
    dist_102 = t_102

data2 = data2[data2.Leadframe != Leadframe_51]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = 0
for a in range(1,103):
    exec('total_dist = total_dist + dist_' + str(a))
print("[51]", "\t", "die_51 = ", die_51, "\t", "Leadframe_51 = ", Leadframe_51, "\t", "die_52 = ", die_52, "\t", "total_dist = ", total_dist)
"""
===============================================
[52] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_52_info = data2[data2.Chip  == die_52]

f_103 =  int(sorted(list(data2[data2['Chip']  == die_52].Distance))[0])
dist_101_info = sorted(list(data2[data2['Chip']  == die_52].Distance))

Leadframe_52 =  list(die_52_info[die_52_info.Distance == f_103].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_52 = list(data2[data2['Chip']  == die_52][data2[data2['Chip']  == die_52].Distance == f_103].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_52].Distance))[0]
dist_101_info_m = sorted(list(data2[data2['Chip']  == die_52].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_101_info_m, count)
ran = map(int, ran)
for t_103 in ran:
    t_103 = t_103

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_103 = f_103
    else:
        dist_103 = t_103                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_52].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_104 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_52].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_52].Distance)))))[0]
dist_104_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_52].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_53 = list(data2[data2['Leadframe']  == Leadframe_52][data2[data2['Leadframe']  == Leadframe_52].Distance == f_104].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_104_info_m, count)
ran = map(int, ran)
for t_104 in ran:
    t_104 = t_104

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_104 = f_104
else:
    dist_104 = t_104

data2 = data2[data2.Leadframe != Leadframe_52]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = 0
for a in range(1,105):
    exec('total_dist = total_dist + dist_' + str(a))
print("[52]", "\t", "die_52 = ", die_52, "\t", "Leadframe_52 = ", Leadframe_52, "\t", "die_53 = ", die_53, "\t", "total_dist = ", total_dist)
"""
===============================================
[53] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_53_info = data2[data2.Chip  == die_53]

f_105 =  int(sorted(list(data2[data2['Chip']  == die_53].Distance))[0])
dist_101_info = sorted(list(data2[data2['Chip']  == die_53].Distance))

Leadframe_53 =  list(die_53_info[die_53_info.Distance == f_105].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_53 = list(data2[data2['Chip']  == die_53][data2[data2['Chip']  == die_53].Distance == f_105].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_53].Distance))[0]
dist_101_info_m = sorted(list(data2[data2['Chip']  == die_53].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_101_info_m, count)
ran = map(int, ran)
for t_105 in ran:
    t_105 = t_105

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_105 = f_105
    else:
        dist_105 = t_105                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_53].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_106 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_53].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_53].Distance)))))[0]
dist_106_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_53].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_54 = list(data2[data2['Leadframe']  == Leadframe_53][data2[data2['Leadframe']  == Leadframe_53].Distance == f_106].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_106_info_m, count)
ran = map(int, ran)
for t_106 in ran:
    t_106 = t_106

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_106 = f_106
else:
    dist_106 = t_106

data2 = data2[data2.Leadframe != Leadframe_53]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = 0
for a in range(1,107):
    exec('total_dist = total_dist + dist_' + str(a))
print("[53]", "\t", "die_53 = ", die_53, "\t", "Leadframe_53 = ", Leadframe_53, "\t", "die_54 = ", die_54, "\t", "total_dist = ", total_dist)

"""
===============================================
[54] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_54_info = data2[data2.Chip  == die_54]

f_107 =  int(sorted(list(data2[data2['Chip']  == die_54].Distance))[0])
dist_101_info = sorted(list(data2[data2['Chip']  == die_54].Distance))

Leadframe_54 =  list(die_54_info[die_54_info.Distance == f_107].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_54 = list(data2[data2['Chip']  == die_54][data2[data2['Chip']  == die_54].Distance == f_107].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_54].Distance))[0]
dist_101_info_m = sorted(list(data2[data2['Chip']  == die_54].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_101_info_m, count)
ran = map(int, ran)
for t_107 in ran:
    t_107 = t_107

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_107 = f_107
    else:
        dist_107 = t_107                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_54].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_108 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_54].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_54].Distance)))))[0]
dist_108_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_54].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_55 = list(data2[data2['Leadframe']  == Leadframe_54][data2[data2['Leadframe']  == Leadframe_54].Distance == f_108].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_108_info_m, count)
ran = map(int, ran)
for t_108 in ran:
    t_108 = t_108

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_108 = f_108
else:
    dist_108 = t_108

data2 = data2[data2.Leadframe != Leadframe_54]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = 0
for a in range(1,109):
    exec('total_dist = total_dist + dist_' + str(a))
print("[54]", "\t", "die_54 = ", die_54, "\t", "Leadframe_54 = ", Leadframe_54, "\t", "die_55 = ", die_55, "\t", "total_dist = ", total_dist)
"""
===============================================
[55] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_55_info = data2[data2.Chip  == die_55]

f_109 =  int(sorted(list(data2[data2['Chip']  == die_55].Distance))[0])
dist_101_info = sorted(list(data2[data2['Chip']  == die_55].Distance))

Leadframe_55 =  list(die_55_info[die_55_info.Distance == f_109].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_55 = list(data2[data2['Chip']  == die_55][data2[data2['Chip']  == die_55].Distance == f_109].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_55].Distance))[0]
dist_101_info_m = sorted(list(data2[data2['Chip']  == die_55].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_101_info_m, count)
ran = map(int, ran)
for t_109 in ran:
    t_109 = t_109

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_109 = f_109
    else:
        dist_109 = t_109                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_55].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_110 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_55].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_55].Distance)))))[0]
dist_110_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_55].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_56 = list(data2[data2['Leadframe']  == Leadframe_55][data2[data2['Leadframe']  == Leadframe_55].Distance == f_110].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_110_info_m, count)
ran = map(int, ran)
for t_110 in ran:
    t_110 = t_110

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_110 = f_110
else:
    dist_110 = t_110

data2 = data2[data2.Leadframe != Leadframe_55]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = 0
for a in range(1,111):
    exec('total_dist = total_dist + dist_' + str(a))
print("[55]", "\t", "die_55 = ", die_55, "\t", "Leadframe_55 = ", Leadframe_55, "\t", "die_56 = ", die_56, "\t", "total_dist = ", total_dist)
"""
===============================================
[56] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_56_info = data2[data2.Chip  == die_56]

f_111 =  int(sorted(list(data2[data2['Chip']  == die_56].Distance))[0])
dist_101_info = sorted(list(data2[data2['Chip']  == die_56].Distance))

Leadframe_56 =  list(die_56_info[die_56_info.Distance == f_111].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_56 = list(data2[data2['Chip']  == die_56][data2[data2['Chip']  == die_56].Distance == f_111].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_56].Distance))[0]
dist_101_info_m = sorted(list(data2[data2['Chip']  == die_56].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_101_info_m, count)
ran = map(int, ran)
for t_111 in ran:
    t_111 = t_111

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_111 = f_111
    else:
        dist_111 = t_111                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_56].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_112 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_56].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_56].Distance)))))[0]
dist_112_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_56].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_57 = list(data2[data2['Leadframe']  == Leadframe_56][data2[data2['Leadframe']  == Leadframe_56].Distance == f_112].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_112_info_m, count)
ran = map(int, ran)
for t_112 in ran:
    t_112 = t_112

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_112 = f_112
else:
    dist_112 = t_112

data2 = data2[data2.Leadframe != Leadframe_56]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = 0
for a in range(1,113):
    exec('total_dist = total_dist + dist_' + str(a))
print("[56]", "\t", "die_56 = ", die_56, "\t", "Leadframe_56 = ", Leadframe_56, "\t", "die_57 = ", die_57, "\t", "total_dist = ", total_dist)
"""
===============================================
[57] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_57_info = data2[data2.Chip  == die_57]

f_113 =  int(sorted(list(data2[data2['Chip']  == die_57].Distance))[0])
dist_101_info = sorted(list(data2[data2['Chip']  == die_57].Distance))

Leadframe_57 =  list(die_57_info[die_57_info.Distance == f_113].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_57 = list(data2[data2['Chip']  == die_57][data2[data2['Chip']  == die_57].Distance == f_113].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_57].Distance))[0]
dist_101_info_m = sorted(list(data2[data2['Chip']  == die_57].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_101_info_m, count)
ran = map(int, ran)
for t_113 in ran:
    t_113 = t_113

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_113 = f_113
    else:
        dist_113 = t_113                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_57].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_114 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_57].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_57].Distance)))))[0]
dist_114_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_57].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_58 = list(data2[data2['Leadframe']  == Leadframe_57][data2[data2['Leadframe']  == Leadframe_57].Distance == f_114].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_114_info_m, count)
ran = map(int, ran)
for t_114 in ran:
    t_114 = t_114

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_114 = f_114
else:
    dist_114 = t_114

data2 = data2[data2.Leadframe != Leadframe_57]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = 0
for a in range(1,115):
    exec('total_dist = total_dist + dist_' + str(a))
print("[57]", "\t", "die_57 = ", die_57, "\t", "Leadframe_57 = ", Leadframe_57, "\t", "die_58 = ", die_58, "\t", "total_dist = ", total_dist)
"""
===============================================
[58] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_58_info = data2[data2.Chip  == die_58]

f_115 =  int(sorted(list(data2[data2['Chip']  == die_58].Distance))[0])
dist_101_info = sorted(list(data2[data2['Chip']  == die_58].Distance))

Leadframe_58 =  list(die_58_info[die_58_info.Distance == f_115].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_58 = list(data2[data2['Chip']  == die_58][data2[data2['Chip']  == die_58].Distance == f_115].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_58].Distance))[0]
dist_101_info_m = sorted(list(data2[data2['Chip']  == die_58].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_101_info_m, count)
ran = map(int, ran)
for t_115 in ran:
    t_115 = t_115

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_115 = f_115
    else:
        dist_115 = t_115                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_58].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_116 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_58].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_58].Distance)))))[0]
dist_116_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_58].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_59 = list(data2[data2['Leadframe']  == Leadframe_58][data2[data2['Leadframe']  == Leadframe_58].Distance == f_116].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_116_info_m, count)
ran = map(int, ran)
for t_116 in ran:
    t_116 = t_116

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_116 = f_116
else:
    dist_116 = t_116

data2 = data2[data2.Leadframe != Leadframe_58]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = 0
for a in range(1,117):
    exec('total_dist = total_dist + dist_' + str(a))
print("[58]", "\t", "die_58 = ", die_58, "\t", "Leadframe_58 = ", Leadframe_58, "\t", "die_59 = ", die_59, "\t", "total_dist = ", total_dist)
"""
===============================================
[59] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_59_info = data2[data2.Chip  == die_59]

f_117 =  int(sorted(list(data2[data2['Chip']  == die_59].Distance))[0])
dist_101_info = sorted(list(data2[data2['Chip']  == die_59].Distance))

Leadframe_59 =  list(die_59_info[die_59_info.Distance == f_117].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_59 = list(data2[data2['Chip']  == die_59][data2[data2['Chip']  == die_59].Distance == f_117].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_59].Distance))[0]
dist_101_info_m = sorted(list(data2[data2['Chip']  == die_59].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_101_info_m, count)
ran = map(int, ran)
for t_117 in ran:
    t_117 = t_117

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_117 = f_117
    else:
        dist_117 = t_117                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_59].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_118 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_59].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_59].Distance)))))[0]
dist_118_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_59].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_60 = list(data2[data2['Leadframe']  == Leadframe_59][data2[data2['Leadframe']  == Leadframe_59].Distance == f_118].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_118_info_m, count)
ran = map(int, ran)
for t_118 in ran:
    t_118 = t_118

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_118 = f_118
else:
    dist_118 = t_118

data2 = data2[data2.Leadframe != Leadframe_59]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = 0
for a in range(1,119):
    exec('total_dist = total_dist + dist_' + str(a))
print("[59]", "\t", "die_59 = ", die_59, "\t", "Leadframe_59 = ", Leadframe_59, "\t", "die_60 = ", die_60, "\t", "total_dist = ", total_dist)
"""
===============================================
[60] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_60_info = data2[data2.Chip  == die_60]

f_119 =  int(sorted(list(data2[data2['Chip']  == die_60].Distance))[0])
dist_101_info = sorted(list(data2[data2['Chip']  == die_60].Distance))

Leadframe_60 =  list(die_60_info[die_60_info.Distance == f_119].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_60 = list(data2[data2['Chip']  == die_60][data2[data2['Chip']  == die_60].Distance == f_119].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_60].Distance))[0]
dist_101_info_m = sorted(list(data2[data2['Chip']  == die_60].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_101_info_m, count)
ran = map(int, ran)
for t_119 in ran:
    t_119 = t_119

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_119 = f_119
    else:
        dist_119 = t_119                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_60].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_120 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_60].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_60].Distance)))))[0]
dist_120_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_60].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_61 = list(data2[data2['Leadframe']  == Leadframe_60][data2[data2['Leadframe']  == Leadframe_60].Distance == f_120].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_120_info_m, count)
ran = map(int, ran)
for t_120 in ran:
    t_120 = t_120

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_120 = f_120
else:
    dist_120 = t_120

data2 = data2[data2.Leadframe != Leadframe_60]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = 0
for a in range(1,121):
    exec('total_dist = total_dist + dist_' + str(a))
print("[60]", "\t", "die_60 = ", die_60, "\t", "Leadframe_60 = ", Leadframe_60, "\t", "die_61 = ", die_61, "\t", "total_dist = ", total_dist)
"""
===============================================
[61] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_61_info = data2[data2.Chip  == die_61]

f_121 =  int(sorted(list(data2[data2['Chip']  == die_61].Distance))[0])
dist_101_info = sorted(list(data2[data2['Chip']  == die_61].Distance))

Leadframe_61 =  list(die_61_info[die_61_info.Distance == f_121].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_61 = list(data2[data2['Chip']  == die_61][data2[data2['Chip']  == die_61].Distance == f_121].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_61].Distance))[0]
dist_101_info_m = sorted(list(data2[data2['Chip']  == die_61].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_101_info_m, count)
ran = map(int, ran)
for t_121 in ran:
    t_121 = t_121

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_121 = f_121
    else:
        dist_121 = t_121                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_61].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_122 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_61].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_61].Distance)))))[0]
dist_122_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_61].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_62 = list(data2[data2['Leadframe']  == Leadframe_61][data2[data2['Leadframe']  == Leadframe_61].Distance == f_122].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_122_info_m, count)
ran = map(int, ran)
for t_122 in ran:
    t_122 = t_122

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_122 = f_122
else:
    dist_122 = t_122

data2 = data2[data2.Leadframe != Leadframe_61]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = 0
for a in range(1,123):
    exec('total_dist = total_dist + dist_' + str(a))
print("[61]", "\t", "die_61 = ", die_61, "\t", "Leadframe_61 = ", Leadframe_61, "\t", "die_62 = ", die_62, "\t", "total_dist = ", total_dist)
"""
===============================================
[62] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_62_info = data2[data2.Chip  == die_62]

f_123 =  int(sorted(list(data2[data2['Chip']  == die_62].Distance))[0])
dist_101_info = sorted(list(data2[data2['Chip']  == die_62].Distance))

Leadframe_62 =  list(die_62_info[die_62_info.Distance == f_123].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_62 = list(data2[data2['Chip']  == die_62][data2[data2['Chip']  == die_62].Distance == f_123].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_62].Distance))[0]
dist_101_info_m = sorted(list(data2[data2['Chip']  == die_62].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_101_info_m, count)
ran = map(int, ran)
for t_123 in ran:
    t_123 = t_123

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_123 = f_123
    else:
        dist_123 = t_123                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_62].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_124 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_62].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_62].Distance)))))[0]
dist_124_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_62].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_63 = list(data2[data2['Leadframe']  == Leadframe_62][data2[data2['Leadframe']  == Leadframe_62].Distance == f_124].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_124_info_m, count)
ran = map(int, ran)
for t_124 in ran:
    t_124 = t_124

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_124 = f_124
else:
    dist_124 = t_124

data2 = data2[data2.Leadframe != Leadframe_62]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = 0
for a in range(1,125):
    exec('total_dist = total_dist + dist_' + str(a))
print("[62]", "\t", "die_62 = ", die_62, "\t", "Leadframe_62 = ", Leadframe_62, "\t", "die_63 = ", die_63, "\t", "total_dist = ", total_dist)
"""
===============================================
[63] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_63_info = data2[data2.Chip  == die_63]

f_125 =  int(sorted(list(data2[data2['Chip']  == die_63].Distance))[0])
dist_101_info = sorted(list(data2[data2['Chip']  == die_63].Distance))

Leadframe_63 =  list(die_63_info[die_63_info.Distance == f_125].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_63 = list(data2[data2['Chip']  == die_63][data2[data2['Chip']  == die_63].Distance == f_125].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_63].Distance))[0]
dist_101_info_m = sorted(list(data2[data2['Chip']  == die_63].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_101_info_m, count)
ran = map(int, ran)
for t_125 in ran:
    t_125 = t_125

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_125 = f_125
    else:
        dist_125 = t_125                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_63].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_126 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_63].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_63].Distance)))))[0]
dist_126_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_63].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_64 = list(data2[data2['Leadframe']  == Leadframe_63][data2[data2['Leadframe']  == Leadframe_63].Distance == f_126].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_126_info_m, count)
ran = map(int, ran)
for t_126 in ran:
    t_126 = t_126

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_126 = f_126
else:
    dist_126 = t_126

data2 = data2[data2.Leadframe != Leadframe_63]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = 0
for a in range(1,127):
    exec('total_dist = total_dist + dist_' + str(a))
print("[63]", "\t", "die_63 = ", die_63, "\t", "Leadframe_63 = ", Leadframe_63, "\t", "die_64 = ", die_64, "\t", "total_dist = ", total_dist)
"""
===============================================
[64] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_64_info = data2[data2.Chip  == die_64]

f_127 =  int(sorted(list(data2[data2['Chip']  == die_64].Distance))[0])
dist_101_info = sorted(list(data2[data2['Chip']  == die_64].Distance))

Leadframe_64 =  list(die_64_info[die_64_info.Distance == f_127].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_64 = list(data2[data2['Chip']  == die_64][data2[data2['Chip']  == die_64].Distance == f_127].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_64].Distance))[0]
dist_101_info_m = sorted(list(data2[data2['Chip']  == die_64].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_101_info_m, count)
ran = map(int, ran)
for t_127 in ran:
    t_127 = t_127

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_127 = f_127
    else:
        dist_127 = t_127                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_64].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_128 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_64].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_64].Distance)))))[0]
dist_128_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_64].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_65 = list(data2[data2['Leadframe']  == Leadframe_64][data2[data2['Leadframe']  == Leadframe_64].Distance == f_128].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_128_info_m, count)
ran = map(int, ran)
for t_128 in ran:
    t_128 = t_128

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_128 = f_128
else:
    dist_128 = t_128

data2 = data2[data2.Leadframe != Leadframe_64]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = 0
for a in range(1,129):
    exec('total_dist = total_dist + dist_' + str(a))
print("[64]", "\t", "die_64 = ", die_64, "\t", "Leadframe_64 = ", Leadframe_64, "\t", "die_65 = ", die_65, "\t", "total_dist = ", total_dist)
"""
===============================================
[65] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_65_info = data2[data2.Chip  == die_65]

f_129 =  int(sorted(list(data2[data2['Chip']  == die_65].Distance))[0])
dist_101_info = sorted(list(data2[data2['Chip']  == die_65].Distance))

Leadframe_65 =  list(die_65_info[die_65_info.Distance == f_129].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_65 = list(data2[data2['Chip']  == die_65][data2[data2['Chip']  == die_65].Distance == f_129].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_65].Distance))[0]
dist_101_info_m = sorted(list(data2[data2['Chip']  == die_65].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_101_info_m, count)
ran = map(int, ran)
for t_129 in ran:
    t_129 = t_129

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_129 = f_129
    else:
        dist_129 = t_129                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_65].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_130 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_65].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_65].Distance)))))[0]
dist_130_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_65].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_66 = list(data2[data2['Leadframe']  == Leadframe_65][data2[data2['Leadframe']  == Leadframe_65].Distance == f_130].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_130_info_m, count)
ran = map(int, ran)
for t_130 in ran:
    t_130 = t_130

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_130 = f_130
else:
    dist_130 = t_130

data2 = data2[data2.Leadframe != Leadframe_65]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = 0
for a in range(1,131):
    exec('total_dist = total_dist + dist_' + str(a))
print("[65]", "\t", "die_65 = ", die_65, "\t", "Leadframe_65 = ", Leadframe_65, "\t", "die_66 = ", die_66, "\t", "total_dist = ", total_dist)
"""
===============================================
[66] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_66_info = data2[data2.Chip  == die_66]

f_131 =  int(sorted(list(data2[data2['Chip']  == die_66].Distance))[0])
dist_101_info = sorted(list(data2[data2['Chip']  == die_66].Distance))

Leadframe_66 =  list(die_66_info[die_66_info.Distance == f_131].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_66 = list(data2[data2['Chip']  == die_66][data2[data2['Chip']  == die_66].Distance == f_131].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_66].Distance))[0]
dist_101_info_m = sorted(list(data2[data2['Chip']  == die_66].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_101_info_m, count)
ran = map(int, ran)
for t_131 in ran:
    t_131 = t_131

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_131 = f_131
    else:
        dist_131 = t_131                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_66].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_132 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_66].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_66].Distance)))))[0]
dist_132_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_66].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_67 = list(data2[data2['Leadframe']  == Leadframe_66][data2[data2['Leadframe']  == Leadframe_66].Distance == f_132].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_132_info_m, count)
ran = map(int, ran)
for t_132 in ran:
    t_132 = t_132

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_132 = f_132
else:
    dist_132 = t_132

data2 = data2[data2.Leadframe != Leadframe_66]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = 0
for a in range(1,133):
    exec('total_dist = total_dist + dist_' + str(a))
print("[66]", "\t", "die_66 = ", die_66, "\t", "Leadframe_66 = ", Leadframe_66, "\t", "die_67 = ", die_67, "\t", "total_dist = ", total_dist)
"""
===============================================
[67] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_67_info = data2[data2.Chip  == die_67]

f_133 =  int(sorted(list(data2[data2['Chip']  == die_67].Distance))[0])
dist_101_info = sorted(list(data2[data2['Chip']  == die_67].Distance))

Leadframe_67 =  list(die_67_info[die_67_info.Distance == f_133].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_67 = list(data2[data2['Chip']  == die_67][data2[data2['Chip']  == die_67].Distance == f_133].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_67].Distance))[0]
dist_101_info_m = sorted(list(data2[data2['Chip']  == die_67].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_101_info_m, count)
ran = map(int, ran)
for t_133 in ran:
    t_133 = t_133

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_133 = f_133
    else:
        dist_133 = t_133                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_67].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_134 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_67].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_67].Distance)))))[0]
dist_134_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_67].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_68 = list(data2[data2['Leadframe']  == Leadframe_67][data2[data2['Leadframe']  == Leadframe_67].Distance == f_134].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_134_info_m, count)
ran = map(int, ran)
for t_134 in ran:
    t_134 = t_134

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_134 = f_134
else:
    dist_134 = t_134

data2 = data2[data2.Leadframe != Leadframe_67]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = 0
for a in range(1,135):
    exec('total_dist = total_dist + dist_' + str(a))
print("[67]", "\t", "die_67 = ", die_67, "\t", "Leadframe_67 = ", Leadframe_67, "\t", "die_68 = ", die_68, "\t", "total_dist = ", total_dist)
"""
===============================================
[68] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_68_info = data2[data2.Chip  == die_68]

f_135 =  int(sorted(list(data2[data2['Chip']  == die_68].Distance))[0])
dist_101_info = sorted(list(data2[data2['Chip']  == die_68].Distance))

Leadframe_68 =  list(die_68_info[die_68_info.Distance == f_135].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_68 = list(data2[data2['Chip']  == die_68][data2[data2['Chip']  == die_68].Distance == f_135].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_68].Distance))[0]
dist_101_info_m = sorted(list(data2[data2['Chip']  == die_68].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_101_info_m, count)
ran = map(int, ran)
for t_135 in ran:
    t_135 = t_135

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_135 = f_135
    else:
        dist_135 = t_135                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_68].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_136 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_68].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_68].Distance)))))[0]
dist_136_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_68].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_69 = list(data2[data2['Leadframe']  == Leadframe_68][data2[data2['Leadframe']  == Leadframe_68].Distance == f_136].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_136_info_m, count)
ran = map(int, ran)
for t_136 in ran:
    t_136 = t_136

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_136 = f_136
else:
    dist_136 = t_136

data2 = data2[data2.Leadframe != Leadframe_68]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = 0
for a in range(1,137):
    exec('total_dist = total_dist + dist_' + str(a))
print("[68]", "\t", "die_68 = ", die_68, "\t", "Leadframe_68 = ", Leadframe_68, "\t", "die_69 = ", die_69, "\t", "total_dist = ", total_dist)
"""
===============================================
[69] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_69_info = data2[data2.Chip  == die_69]

f_137 =  int(sorted(list(data2[data2['Chip']  == die_69].Distance))[0])
dist_101_info = sorted(list(data2[data2['Chip']  == die_69].Distance))

Leadframe_69 =  list(die_69_info[die_69_info.Distance == f_137].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_69 = list(data2[data2['Chip']  == die_69][data2[data2['Chip']  == die_69].Distance == f_137].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_69].Distance))[0]
dist_101_info_m = sorted(list(data2[data2['Chip']  == die_69].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_101_info_m, count)
ran = map(int, ran)
for t_137 in ran:
    t_137 = t_137

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_137 = f_137
    else:
        dist_137 = t_137                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_69].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_138 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_69].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_69].Distance)))))[0]
dist_138_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_69].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_70 = list(data2[data2['Leadframe']  == Leadframe_69][data2[data2['Leadframe']  == Leadframe_69].Distance == f_138].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_138_info_m, count)
ran = map(int, ran)
for t_138 in ran:
    t_138 = t_138

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_138 = f_138
else:
    dist_138 = t_138

data2 = data2[data2.Leadframe != Leadframe_69]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = 0
for a in range(1,139):
    exec('total_dist = total_dist + dist_' + str(a))
print("[69]", "\t", "die_69 = ", die_69, "\t", "Leadframe_69 = ", Leadframe_69, "\t", "die_70 = ", die_70, "\t", "total_dist = ", total_dist)
"""
===============================================
[70] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_70_info = data2[data2.Chip  == die_70]

f_139 =  int(sorted(list(data2[data2['Chip']  == die_70].Distance))[0])
dist_101_info = sorted(list(data2[data2['Chip']  == die_70].Distance))

Leadframe_69 =  list(die_70_info[die_70_info.Distance == f_139].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_70 = list(data2[data2['Chip']  == die_70][data2[data2['Chip']  == die_70].Distance == f_139].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_70].Distance))[0]
dist_101_info_m = sorted(list(data2[data2['Chip']  == die_70].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_101_info_m, count)
ran = map(int, ran)
for t_139 in ran:
    t_139 = t_139

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_139 = f_139
    else:
        dist_139 = t_139                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_70].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_140 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_70].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_70].Distance)))))[0]
dist_140_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_70].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_71 = list(data2[data2['Leadframe']  == Leadframe_70][data2[data2['Leadframe']  == Leadframe_70].Distance == f_140].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_140_info_m, count)
ran = map(int, ran)
for t_140 in ran:
    t_140 = t_140

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_140 = f_140
else:
    dist_140 = t_140

data2 = data2[data2.Leadframe != Leadframe_70]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = 0
for a in range(1,141):
    exec('total_dist = total_dist + dist_' + str(a))
print("[70]", "\t", "die_70 = ", die_70, "\t", "Leadframe_70 = ", Leadframe_70, "\t", "die_71 = ", die_71, "\t", "total_dist = ", total_dist)
"""
===============================================
[71] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_71_info = data2[data2.Chip  == die_71]

f_141 =  int(sorted(list(data2[data2['Chip']  == die_71].Distance))[0])
dist_101_info = sorted(list(data2[data2['Chip']  == die_71].Distance))

Leadframe_69 =  list(die_71_info[die_71_info.Distance == f_141].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_71 = list(data2[data2['Chip']  == die_71][data2[data2['Chip']  == die_71].Distance == f_141].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_71].Distance))[0]
dist_101_info_m = sorted(list(data2[data2['Chip']  == die_71].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_101_info_m, count)
ran = map(int, ran)
for t_141 in ran:
    t_141 = t_141

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_141 = f_141
    else:
        dist_141 = t_141                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_71].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_142 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_71].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_71].Distance)))))[0]
dist_142_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_71].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_72 = list(data2[data2['Leadframe']  == Leadframe_71][data2[data2['Leadframe']  == Leadframe_71].Distance == f_142].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_142_info_m, count)
ran = map(int, ran)
for t_142 in ran:
    t_142 = t_142

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_142 = f_142
else:
    dist_142 = t_142

data2 = data2[data2.Leadframe != Leadframe_71]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = 0
for a in range(1,143):
    exec('total_dist = total_dist + dist_' + str(a))
print("[71]", "\t", "die_71 = ", die_71, "\t", "Leadframe_71 = ", Leadframe_71, "\t", "die_72 = ", die_72, "\t", "total_dist = ", total_dist)
"""
===============================================
[72] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_72_info = data2[data2.Chip  == die_72]

f_143 =  int(sorted(list(data2[data2['Chip']  == die_72].Distance))[0])
dist_101_info = sorted(list(data2[data2['Chip']  == die_72].Distance))

Leadframe_72 =  list(die_72_info[die_72_info.Distance == f_143].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_72 = list(data2[data2['Chip']  == die_72][data2[data2['Chip']  == die_72].Distance == f_143].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_72].Distance))[0]
dist_101_info_m = sorted(list(data2[data2['Chip']  == die_72].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_101_info_m, count)
ran = map(int, ran)
for t_143 in ran:
    t_143 = t_143

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_143 = f_143
    else:
        dist_143 = t_143                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_72].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_144 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_72].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_72].Distance)))))[0]
dist_144_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_72].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_73 = list(data2[data2['Leadframe']  == Leadframe_72][data2[data2['Leadframe']  == Leadframe_72].Distance == f_144].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_144_info_m, count)
ran = map(int, ran)
for t_144 in ran:
    t_144 = t_144

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_144 = f_144
else:
    dist_144 = t_144

data2 = data2[data2.Leadframe != Leadframe_72]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = 0
for a in range(1,145):
    exec('total_dist = total_dist + dist_' + str(a))
print("[72]", "\t", "die_72 = ", die_72, "\t", "Leadframe_72 = ", Leadframe_72, "\t", "die_73 = ", die_73, "\t", "total_dist = ", total_dist)
"""
===============================================
[73] 웨이퍼에서 리드프레임으로, 리드프레임에서 웨이퍼로 옮기는 알고리즘 설계~~~~~~~~~~~~~새로운 리드프레임으로 교체

# die 5 (i) -> leadframe 5 , leadframe 5(j) -> die 6(i+1),,,, f= 9 t=10, dist = 5,6 /// l = 7, k = 8
===============================================
"""

die_73_info = data2[data2.Chip  == die_73]

f_145 =  int(sorted(list(data2[data2['Chip']  == die_73].Distance))[0])
dist_101_info = sorted(list(data2[data2['Chip']  == die_73].Distance))

Leadframe_73 =  list(die_73_info[die_73_info.Distance == f_145].Leadframe )[0]

del dist_1_info[0]
dist_1_info_m= dist_1_info

Leadframe_73 = list(data2[data2['Chip']  == die_73][data2[data2['Chip']  == die_73].Distance == f_145].Leadframe)[0]      # 길이가 같아면 숫자가 작은 쪽을 선택

del sorted(list(data2[data2['Chip']  == die_73].Distance))[0]
dist_101_info_m = sorted(list(data2[data2['Chip']  == die_73].Distance))

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_101_info_m, count)
ran = map(int, ran)
for t_145 in ran:
    t_145 = t_145

for i in range(1):
    epsilon = random.random()

    if epsilon > 0.2:
        dist_145 = f_145
    else:
        dist_145 = t_145                                                                                                     # 여기까지가die_1 골라서Leaframe_1 까지e-greedy 적용.


data2 = data2.drop(data2[data2.Chip == die_73].index)                                # 옮겨진die1에 해당하는 애는 삭제

f_146 = int( sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_73].Distance)))))[0])

del sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_73].Distance)))))[0]
dist_146_info_m = list(set(sorted(list(set(sorted(list(data2[data2['Leadframe']  == Leadframe_73].Distance)))))))                                                                                                     # i 는 거리 최소값 뺀 거리 리스트)

die_74 = list(data2[data2['Leadframe']  == Leadframe_73][data2[data2['Leadframe']  == Leadframe_73].Distance == f_146].Chip)[0]  # 길이가 같아지면 숫자가 작은 쪽을 선택

count = 1                                                                                                          # 1 - epsilon 인 것은 최소값 뺀 리스트 형성한뒤, 그중에서 값 랜덤하게 추출해서 정수형으로 반환
ran = random.sample(dist_146_info_m, count)
ran = map(int, ran)
for t_146 in ran:
    t_146 = t_146

for i in range(1):
    epsilon = random.random()
if epsilon > 0.2:
    dist_146 = f_146
else:
    dist_146 = t_146

data2 = data2[data2.Leadframe != Leadframe_73]                                       # 옮겨진Leadframe1에 해당하는 애는 삭제
total_dist = 0
for a in range(1,147):
    exec('total_dist = total_dist + dist_' + str(a))
print("[73]", "\t", "die_73 = ", die_73, "\t", "Leadframe_73 = ", Leadframe_73, "\t", "die_74 = ", die_74, "\t", "total_dist = ", total_dist)
