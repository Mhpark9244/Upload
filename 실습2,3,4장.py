"""
#예제 2-1
a = "you need python"
b = a[0] + a[1]+ a[2]
print(b)

c = a[0:3]
print(c)

d= a[:3]
print(d)

#예제 2-2
a = " I eat %d apples" %3
print(a)

#예제 2-3
a = " I eat %d apples" %3
print(a)

#예제 2-4
number =3
a = "I eat %d apples" % number
print(a)

#예제 2-5
number = 10
day = "three"
a = "I ate %d appeles. So I was sick for %s days" %(number, day)
print(a)

#예제 2-6
a = "I ate {number} apples. So I was work for {day} days." .format(number = 10, day =3)
print(a)

#예제 2-7
dict = {'Name' : 'pey','Phone':'011999124', 'Birth' : '991209' }
print(dict['Name'])
print(dict.keys())
print(dict.values())
print(dict.items())
#===========================================================================================

#예제 3-1
money = 1
if money :
    print("택시를 타고가라")
else:
    print("걸어 가라")

#예제 3-2
money = 2000

if money >= 3000:
    print("택시를 타고가라")
else:
    print("걸어 가라")

#예제 3-3
money = 2000
card = 1
if money >= 3000 or card:
    print("택시를 타고가라")
else:
    print("걸어 가라")

#예제 3-4
money = 5000
card = False
if money >= 4000 or card:
    print("택시를 탈 수 있다.")
else:
    print("택시를 탈 수 없다.")

#예제 3-5
pocket = ['paper', 'cellphone', 'money']
if 'money' in pocket:
    print("택시를 타고 가라.")
else:
    print("걸어가라")

#예제 3-6
pocket = ['paper','cellphone']
card = 1
if 'money' in pocket:
    print("택시를 타고 가라")
elif card :
    print("택시를 타고 가라")
else:
    print("걸어가라")

#예제 3-7
treehit = 0
while treehit < 10:
    treehit = treehit +1
    print("나무를 %d번 찍었습니다." % treehit)
    if treehit == 10:
        print("나무 넘어갑니다.")

#예제 3-8
coffee = 5
while True:
    money = int(input("돈을 넣어주세요 : "))
    if money == 300:
        print("커피를 줍니다.")
        coffee = coffee -1
        print("남은 커피의 양은 %d개 입니다." % coffee)
    elif money > 300 :
        print("거스름돈 %d를 주고 커피를 줍니다." % (money - 300))
        coffee = coffee -1
        print("남은 커피의 양은 %d개 입니다." % coffee)
    else:
        print("돈을 다시 돌려주고 커피를 주지 않습니다.")
        print("남은 커피의 양은 %d개입니다." % coffee)
    if not coffee:
        print("커피가 다 떨어졌습니다. 판매를 중지합니다.")
        break

# 예제3-9
a =0
while a < 10:
    a = a+1
    if a % 2 == 0 : continue
    print(a)

#예제 3-10
a =0
while True:
    a = a +1
    if a > 5 :break
    print('*' * a)

# 예제 3-11
a = 0
while True:
    a = a+1
    if a > 4 :break
    print('*' * a + ' '* a + '*' * a)

# 예제 3-12
a = [(1,2),(3,4),(5,6)]
for (first, last) in a:
    print(first+last)
"""
#예제 3-13
scorelist = [90,40,70,80,65]
a = 0
for score in scorelist :
    a = a +1
    if score >= 80:
        print("%d번 학생은 합격입니다." %a)
    else:
        print("%d번 학생은 불합격입니다." %a)
"""
#예제 3-14
scorelist = [90,40,70,80,65]
a = 0
for score in scorelist:
    a = a +1
    if score < 80:continue
    print("%d번 학생은 합격입니다." %a)

#예제 3-15
sum = 0
for i in range(1,11):
    sum = sum + i
print(sum)

#예제 3-16
for i in range(1,10):
    for j in range(1,10):
        print(i*j, end = " ")
    print('')

#===========================================================================================
#예제4-1
def sum(a,b):
    return a+b
def linear(a,b):
    return 4*a+3*b+20

a = 3
b = 4
c = sum(a,b)
d = linear(a,b)

print(c)
print(d)

#예제 4-2
def sum_many(*args) :
	sum = 0
	for i in args:
		sum = sum + i
	return sum
a =60
b = 40
c = 20

result = sum_many(a,b,c)
print(result)

#예제 4-3
number = input("숫자를 입력하세요 :")

#예제 4-4)
for i in range(10):
    #print(i)
     print (i, end = '')
"""
