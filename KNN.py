from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# k-近邻算法

def classify0(inX, dataSet, labels, k):
    # 列出数组长度
    dataSetSize = dataSet.shape[0]
    # 算出离偏移量 的x，y轴距离
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # 对 x y 分别求平方
    sqDiffMat = diffMat ** 2
    # 将x y 求和
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方算出距离 即 d = /x2 + y2
    distances = sqDistances ** 0.5
    # 返回距离从小到大的索引值
    sortedDistIndicies = distances.argsort()
    classCount = {}

    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # get方法返回classCount中键名对应的值 没有则返回默认值
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 根据键值进行降序排列
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


group, labels = createDataSet()

print(classify0((0, 0), group, labels, 3))


## 将数据转换成分页器接受的格式

def file2matrix(filename):
    # 读取文件
    fs = open(filename)
    # 将文件内容分成为每一行的列表
    arrayOLines = fs.readlines()
    # 计算出行数
    numberOfLines = len(arrayOLines)
    # 创建一个 空的矩阵
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    # 通过循环 将数据形成 classify0 能接受的格式
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

print(file2matrix('data/datingTestSet2.txt'))