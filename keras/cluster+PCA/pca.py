import numpy as np
import nrrd
from sklearn.decomposition import PCA

Case_num = 77
Threshold = 10


res = np.load(str(Case_num)+'-black.save')
res = res[res[:, 2].argsort()]
print(res.shape)

def label(res):

    start = res[0][2]
    num = [0]
    for i in range(len(res)):
        if res[i][2] != start:
            start = res[i][2]
            num.append(i)
    return num


nrrdData = nrrd.read('Case0%d.nrrd'%Case_num)
im = nrrdData[0]


mask = np.zeros(im.shape)
for coord in res:
    mask[int(coord[0]),int(coord[1]),int(coord[2])]=1.0
nrrd.write('mask%d-black.nrrd'%Case_num, mask, nrrdData[1])


mid = []
rest = res
final = []
while len(rest) > 1:
    flag = 0
    ini = [0]
    cluster = []
    while flag != 1:
        flag = 1
        if cluster == []:
            cluster = rest[[ini]]
        else :
            cluster = np.concatenate((cluster, rest[ini]), axis=0)
        rest = np.delete(rest,ini,0)
        ini = []
        for ele in cluster:
            for i,j in enumerate(rest):
                if j[2] >= ele[2]:
                    dist = abs(ele[0]-j[0])+abs(ele[1]-j[1])+abs(ele[2]-j[2])
                    if dist < 3:
                        if i not in ini:
                            ini.append(i)
                        flag = 0
    if len(cluster) > 1:
        cluster = cluster[cluster[:, 2].argsort()]
        layers = label(cluster)
        if len(layers) > 3:
            X = cluster
            pca = PCA(n_components=2)
            pca.fit(X)
            ratio = pca.explained_variance_ratio_
            print(ratio)
            print(ratio[0]/ratio[1])
            if ratio[0]/ratio[1] > Threshold:
                mid.append(np.average(cluster[:,:3], axis=0, weights=cluster[:,3]))
                if final == []:
                    final = cluster[:,:3]
                else:
                    final = np.concatenate((cluster[:,:3], final), axis=0)
print('done!')
print('tips:', len(mid))
res = final
print(res.shape)

mask = np.zeros(im.shape)
for coord in res:
    mask[int(coord[0]),int(coord[1]),int(coord[2])]=1.0
nrrd.write('mask-latest%d-black.nrrd'%Case_num, mask, nrrdData[1])



f = open(str(Case_num)+'-cluster.save', 'wb')
np.save(f, mid)
f.close