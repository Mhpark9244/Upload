################################# 1번 #################################
print('\n################################# 1번 #################################\n')
import warnings
warnings.filterwarnings(action='ignore')

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

class PFA(object):
    def __init__(self, n_features, q=None):
        self.q = q
        self.n_features = n_features

    def fit(self, X):
        if not self.q:
            self.q = X.shape[1]

        sc = StandardScaler()
        X = sc.fit_transform(X)

        A_q = X.T

        kmeans = KMeans(n_clusters=self.n_features).fit(A_q)
        clusters = kmeans.predict(A_q)
        cluster_centers = kmeans.cluster_centers_

        dists = defaultdict(list)
        for i, c in enumerate(clusters):
            dist = euclidean_distances([A_q[i, :]], [cluster_centers[c, :]])[0][0]
            dists[c].append((i, dist))

        self.indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]
        self.features_ = X[:, self.indices_]

import pandas as pd
import numpy as np

face = pd.read_excel('d:/Facial_expressions.xlsx')

pfa = PFA(n_features=2)
pfa.fit(face)

column_indices = pfa.indices_
f1 = face.columns[column_indices[0]]
f2 = face.columns[column_indices[1]]
print('변수 선택 이유: 위에 정의된 Principal Feature Analysis를 이용해서 변수를 선택하였다.\n'
'PFA class의 작동 과정은 다음과 같다.\n'
'Dataset을 이용해서 표준화 과정을 적용하고, transpose시켜서 기존의 레코드 대신에 원래 dataset의 특징 변수를 레코드로 본다.\n'
'이러한 레코드(원래 dataset의 특징변수)들을 임의로 X라고 할 때, X에 대해서 k-means clustering을 진행한다. 이 때, 군집의 개수는 선택하고 싶은 변수의 개수로 선정한다.\n'
'군집화가 완료되면 각 군집 중심과 동일 군집에 속한 X와의 모든 거리를 계산한 후에 군집 중심과 가장 가까운 X1(군집1의 X)과 X2(군집2의 X)를 변수로 선택한다.\n'
'즉, 기존의 특징 변수들을 레코드로 본 후에 군집화를 실시하고 각 군집을 가장 잘 설명하는 두 개의 특징 변수를 구하는 방법으로 변수를 선택하였다.\n'
'이러한 결과 선택된 변수는 "{0}"와 "{1}"이다.\n'.format(f1, f2))

face_pfa = face.iloc[ : , [column_indices[0], column_indices[1]]]
# X = np.array(face_pfa)
X = pfa.features_

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#sns.relplot(face_pfa.columns[0], face_pfa.columns[1], data=face_pfa)
plt.show()

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import davies_bouldin_score
import matplotlib.cm as cm

for n_clusters in range(2, 8):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(12, 5)

    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    silhouette_avg = silhouette_score(X, cluster_labels)
    db_score = davies_bouldin_score(X, cluster_labels)
    sse = clusterer.inertia_
    print("For n_clusters = {0}\nThe average silhouette_score is : {1}, The davies_bouldin_score is : {2}, Sum of squared error is : {3}\n".format(n_clusters, silhouette_avg, db_score, sse))

    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5*size_cluster_i, str(i))
        ax1.text(0.825, y_lower + 0.5*size_cluster_i, str(np.round(ith_cluster_silhouette_values.mean(),decimals=4)))

        y_lower = y_upper + 10

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

    centers = clusterer.cluster_centers_

    ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the {0}".format(f1))
    ax2.set_ylabel("Feature space for the {0}".format(f2))

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                  fontsize=14, fontweight='bold')
plt.show()

################################# 2번 #################################
print('\n################################# 2번 #################################\n')
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.rcParams["figure.figsize"]= (12, 8)
colors = ['k','m','r','g','c','y','b']

def clustering_and_plot(data, argument, optk):
    model = AgglomerativeClustering(n_clusters=optk)
    model.fit(data)
    cluster_labels = model.fit_predict(data)

    data['labels'] = cluster_labels
    g1 = [np.array(data.groupby('labels').mean())[i].reshape((68,3)) for i in range(optk)]

    for k in range(optk):
        df = pd.DataFrame(g1[k], columns=['x','y','z'])
        df.drop('z', axis=1)
        plt.scatter(df['x'], df['y'], c=colors[k], s=20)
        plt.xlabel('x coordinates')
        plt.ylabel('y coordinates')
        plt.title("x and y coordinates plot for each {0} cluster's centroids - {1}".format(optk, argument), fontsize=14, fontweight='bold')
    plt.show()


def finding_optK(data, range):
    silhouettes = []
    dbscores = []
    K = list(range)
    for k in K:    
        model = AgglomerativeClustering(n_clusters=k)
        model.fit(data)
        cluster_labels = model.fit_predict(data) 

        silhouettes.append(silhouette_score(data, cluster_labels))
        dbscores.append(davies_bouldin_score(data, cluster_labels))

    ax = plt.subplot(111)
    ax.plot(K, silhouettes, 'mo-',label='Silhouette')
    ax.plot(K, dbscores, 'go-',label='DBI')
    ax.legend(loc='upper right')
    ax.set_xlabel('K')
    ax.set_ylabel('Silhouette / DBI')
    plt.title('Silhouette coefficient / Davies-Bouldin index', fontsize=14, fontweight='bold')
    plt.show()


def drawing_dendrogram(data, **options):
    Z = linkage(data.values, method='ward')

    if options.get('n_cluster') == None:
        plt.title("Hierarchical Clustering Dendrogram", fontsize=14, fontweight='bold')
    else:
        plt.title("Hierarchical Clustering Dendrogram with {0} clusters".format(options['n_cluster']), fontsize=14, fontweight='bold')
        if options.get('n_cluster') == 3:
            plt.axhline(y=120, color="k", linestyle="--")
        elif options.get('n_cluster') == 5:
            plt.axhline(y=60, color="k", linestyle="--")
    dendrogram(Z, truncate_mode='lastp')
    plt.show()


face = pd.read_excel('d:/Facial_expressions.xlsx')
X = face.drop(0, axis=1)

sc = StandardScaler()
X1 = pd.DataFrame(sc.fit_transform(X.astype(float)), columns=X.columns)

drawing_dendrogram(X1)
drawing_dendrogram(X1, n_cluster=3)
drawing_dendrogram(X1, n_cluster=5)

finding_optK(X1, range(2,50))
print('k=2일 때가 silhouette score와 davies-bouldin index가 가장 좋지만 SSE가 너무 크기 때문에 제외하고,\n'
'silhouette score가 1에 가깝고 davies-bouldin index가 낮은 k=5가 적절하다고 판단.\n')

print('Hierarchical clusting으로 data를 학습시키고 각각의 군집 중심을 plot 했을 경우 얼굴 형태가 나오는 것을 판단할 수 있다.\n'
'x와 y를 픽셀의 좌표값으로 볼 경우 군집 중심을 보게 되면 픽셀의 위치에 따라 군집이 이루어지는 것을 볼 수 있다.\n'
'그러나 표준화를 하지 않은 data의 경우 한 개의 군집만을 제외하고는 많은 부분 겹쳐 있는 것을 Plot에서 볼 수 있다.\n')
clustering_and_plot(X, 'without Standardization', 5)

print('따라서 표준화를 한 data에 대해 모델을 학습시키고 각 군집 중심을 이용해 plot을 하게 되면,\n'
'표준화의 결과에 의해 얼굴의 형태는 잘 보이지 않지만 군집 중심의 픽셀 위치가 확실하게 구분되는 것을 알 수 있다.\n'
'그러므로 표준화 한 경우에 대해 군집의 결과를 해석해보면\n'
'픽셀의 x좌표가 가장 작고 y좌표가 가장 큰 하늘색 군집1이 생성되고, x좌표가 중간이고 y좌표가 작은 빨간색 군집 1개, x좌표가 큰 편에 속하고 y좌표가 가장 작은 보라색 군집 1개,\n'
'x좌표가 큰 편에 속하고 y좌표는 중간인 초록색 군집 1개, 그리고 x좌표가 가장 크고, y좌표도 큰 편에 속하는 검은색 군집 1개 총 5개의 군집이 형성된다.\n'
'각 군집을 보게 되면 어떠한 군집의 군집 중심은 매우 응축되어서 얼굴 형태가 보이지도 않고, 어떠한 군집의 군집 중심은 꽤 퍼져있는 것을 알 수 있다.\n'
'이러한 결과를 봤을 때 검은색 군집이 군집 내의 레코드가 가장 잘 뭉쳐있다고 판단할 수 있다. ("with-in cluster distance"가 가장 작다)\n'
'반대로 하늘색의 군집을 보았을 때, 표준화를 했음에도 얼굴 형태가 꽤 잘 보일 정도로 군집 내의 레코드가 퍼져있다고 판단할 수 있다. ("with-in cluster distance"가 가장 크다)')
clustering_and_plot(X1, 'with Standardization', 5)