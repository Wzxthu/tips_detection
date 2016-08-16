import numpy as np
import nrrd

res = np.load('full+pro.save')
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

def max_dist(a):
    max = 0
    for num,i in enumerate(a):
        for j in a[num+1:]:
            vec = (i-j)[:3]
            if np.dot(vec,vec) > max:
                max = np.dot(vec,vec)
    return max

ori = res
temp = []
num = label(ori)
for i in range(8, len(num)-10):
    rest = ori[num[i]:num[i+1]]
    print(ori[num[i]][2])
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
                    dist = abs(ele[0]-j[0])+abs(ele[1]-j[1])
                    if dist == 1 or dist == 2 or dist == 3:
                        if i not in ini:
                            ini.append(i)
                        flag = 0
        if len(cluster) > 4 and max_dist(cluster) < 80:
            if final == []:
                final = cluster
            else:
                final = np.concatenate((cluster, final), axis=0)
    if temp == []:
        temp = final
    else:
        temp = np.concatenate((final, temp), axis=0)
    print('done one!')

res = temp
print(res.shape)

nrrdData = nrrd.read('Case064.nrrd')
im = nrrdData[0]

def judge(cluster):
    coord = np.mean(cluster, axis=0)
    if im[int(coord[0]),int(coord[1]),int(coord[2])]<100 and len(cluster)>15:
        return 1
    else:
        return 0


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
                dist = abs(ele[0]-j[0])+abs(ele[1]-j[1])+abs(ele[2]-j[2])
                if dist == 1 or dist == 2:
                    if i not in ini:
                        ini.append(i)
                    flag = 0
    if judge(cluster[:,:3]):
        cluster = cluster[cluster[:, 2].argsort()]
        num = label(cluster)
        if len(num) > 2:
            cluster = cluster[num[-3]:]
            cluster = cluster[cluster[:, 3].argsort()]
            mid.append(cluster[-1,:3])
            # mid.append(np.average(cluster[:,:3], axis=0, weights=cluster[:,3]))
            # mid.append(np.mean(cluster, axis=0))
            if final == []:
                final = cluster[:,:3]
            else:
                final = np.concatenate((cluster[:,:3], final), axis=0)
print('done!')
print('tips:', len(mid))
res = final
print(res.shape)

# mask = np.zeros(im.shape)
# for coord in res:
#     mask[int(coord[0]),int(coord[1]),int(coord[2])]=1.0
# nrrd.write('mask-latest-64-tip4.nrrd', mask, nrrdData[1])


f = open('tips+color+promax.save', 'wb')
np.save(f, mid)
f.close