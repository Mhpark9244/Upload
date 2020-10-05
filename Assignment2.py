
#과제 2-1
a = "20180901Rainy"
date =a[0:8]
weather = a[8:]
print(date, weather)

#과제 2-2
a = "pithon"
print(a.replace(a[1],"y"))

#과제 2-3
a = ["Life", "is", "too", "short", "you", "need", "python"]
print(a[4]+" "+a[2])

#과제 2-4
a = [1,3,5,4,2]
print(a.sort())

b = [1,3,5,4,2]
b.sort(reverse=True)
print(a,b)

#과제 2-5
s1 = set(["a", "b", "c", "d","e"])
s2 = set(["c", "d", "e", "f", "g"])
print(s1-s2)

# 과제 2-6
list = [1,9,23,46]
My_number = int(input("값을 입력하세요 : "))
if My_number in list:
    print("당첨")
else:
    print("꽝")

# 과제 2-7
number = int(input("숫자를 입력하시오:"))
if number % 2 == 0:
    print("짝수")
else:
    print("홀수")

# 과제 2-8
# shirt

# 과제 2-9
star = 7
space = 0
while star > 0:
    print(' ' * space + '*' * star)
    star = star - 2
    space = space + 1

# 과제 2-10
k = 5
for i in range(0, 5):
    k = k - 1
    print("*" * (2 ** int(i + 1)) + " " * k + "+" * (i + 1))

# 과제 2-11
for i in range(1, 101):
    print(i)

#과제 2-12
k = []
for i in range(1,101):
    if i % 5 == 0:
        k.append(i)
print(k)
print(sum(k))

