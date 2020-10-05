"""
#클래스 예제
class FourCal:
    def setdata(self,first,second):
        self.first = first
        self.second = second
    def sum(self):
        result = self.first + self.second
        return result
    def mul(self):
        result = self.first * self.second
        return result
    def sub(self):
        result = self.first - self.second
        return result
    def div(self):
        result = self.first / self.second
        return result

a = FourCal()
a.setdata(4,2)
print(a.sum())
print(a.mul())
print(a.sub())
print(a.div())



#예제 6-1
import pandas as pd
data = [1,3,5,7,9]
s=pd.Series(data)
print(s)

#예제 6-2
import pandas as pd
data = {'year' : [2016, 2017, 2018],
    'GDP rate' : [ 2.8, 3.1, 3.0],
    'GDP' : ['1.6M', '1,7M', '1.8M']}
df =  pd.DataFrame(data)
Dfd = df.describe()
print(df)
print(Dfd)

#예제 6-3
import pandas as pd
import numpy as np

data = np.random.rand(2,3,4)
p = pd.Panel(data)
print(p)

#예제 6-4
import pandas as pd

#df = pd.read_excel('D:/record.xlsx')
df= pd.read_excel('C:/Users\pgk82/Desktop/practice/record.xlsx')
print(df)

#예제 6-5
import numpy as np
list = [1,2,3,4]
a = np.array(list)
b = np.array([[1,2,3],[4,5,6]])

print(b.shape)

#예제 6-6,7,8 통과
#예제 6-9
import numpy as np
lst1 = [
    [1,2],
    [3,4]
]
lst2 = [
    [5,6],
    [7,8]
]
a=np.array(lst1)
b=np.array(lst2)

c = np.dot(a,b)
print(c)

#예제 6-9
import scipy as sp
import numpy as np
from scipy import stats

N =100
theta_0 = 0.35
np.random.seed(0)
x = sp.stats.bernoulli(theta_0).rvs(N)
n = np.count_nonzero(x)
print(n)
y = sp.stats.binom_test(n,N)
print(y)

#예제 6-10
import numpy as np
import scipy as sp

N = 10
k = 4
theta_0 = np.ones(k)/k
x = np.random.seed(0)
x = np.random.choice(k,N,p=theta_0)
n = np.bincount(x,minlength=k)
print(n)
y = sp.stats.chisquare(n)
print(y)

#예제 6-11
import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pylab as plt
import matplotlib as mpl
import seaborn as sns
from seaborn import distplot

np.random.seed(0)
N1 = 50
N2 = 100
x1 = sp.stats.norm(0,1).rvs(N1)
x2 = sp.stats.norm(0.5,1.5).rvs(N2)
sns.distplot(x1)
sns.distplot(x2)
a = plt.show()
y = sp.stats.ks_2samp(x1,x2)
print(a)
print(y)

#exercise
import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pylab as plt
import matplotlib as mpl
import seaborn as sns
from seaborn import distplot

np.random.seed(0)
N1 = 30
N2 = 50
N3 = 100
x1 = sp.stats.norm(0,6).rvs(N1)
x2 = sp.stats.norm(0.5).rvs(N2)
x3 = sp.stats.norm(0.3,0.7).rvs(N3)

sns.distplot(x1)
sns.distplot(x2)
sns.distplot(x3)
a = plt.show()
y1 = sp.stats.ks_2samp(x1,x2)
y2 = sp.stats.ks_2samp(x2,x3)
y3 = sp.stats.ks_2samp(x1,x3)

print(a)
print(y1)
print(y2)
print(y3)

#예제 6-12
from matplotlib import pyplot as plt
import numpy as np

x = np.arange(1,10)
y = 5*x
plt.plot(x,y)
plt.show()

#예제 6-13
from matplotlib import pyplot as plt
import numpy as np

x = np.arange(1,10,0.1)
y = 0.2 * x
y2 = np.sin(x)

plt.plot(x,y,'b', label = 'first')
plt.plot(x,y2,'r', label = 'second')

plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('matplot example')
plt.legend(loc = 'upper right')
plt.show()

#예제 6-14
from matplotlib import pyplot as plt
import numpy as np

x = np.arange(1,10)
y1 = x * 5
y2 = x * x
y3 = x * x * x
y4 = np.sin(x)

plt.subplot(2,2,1)
plt.plot(x,y1)

plt.subplot(2,2,2)
plt.plot(x,y2,'r')

plt.subplot(2,2,3)
plt.plot(x,y3,'g')

plt.subplot(2,2,4)
plt.plot(x,y4,'orange')

plt.show()


#예제6-15
import math
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris

data = load_iris()

print(data)
"""
#예제 6-16 petal_length, petal_width 산점도 그리기
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

data = load_iris()
print(data)

feature = data.data
feature_names = data.feature_names
target = data.target
target_names = data.target_names

for t in range(3):
    if t == 0:
        #c = 'r'
        marker = '>'
    elif t == 1:
        #c = 'g'
        marker = 'x'
    elif t == 2:
        #c = 'b'
        marker = 'o'
    plt.scatter(feature[target== t,2],
                feature[target== t,3],
                marker = marker)
                #c = c)
    plt.xlabel('petal_length')
    plt.ylabel('petal_width')
    
x = np.arange(1,5)
y = -(2/3)*x+2
plt.plot(x,y,'black')

plt.show()

#예제 6-17 Setosa 분류하기 (target = 0)
pl_length = feature[:,2]
labels = target_names[target]
is_setosa = (labels == 'setosa')
max_setosa = pl_length[is_setosa].max()
min_non_setosa = pl_length[~is_setosa].min()
print("Maximum of setosa : {0}".format(max_setosa))
print("Minimum of others : {0}".format(min_non_setosa))

#예제 6-18 분류직선

x = np.arange(1,5)
y = -(2/3)*x+2
plt.plot(x,y,'black')
