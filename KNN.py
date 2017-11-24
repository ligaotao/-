from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt


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


# 将数据转换成分页器接受的格式

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
        # 第 index 行
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# 数据归一化
# 由于 飞行里程的数字偏大 将会影响到计算结果
# 我们需要将它转换为 0-1以内的值
# 转换公式 newValue = (oldValue - min) / max - min

def autoNorm(dataSet):
    print(dataSet)
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals

    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    print(normDataSet)
    normDataSet = normDataSet / tile(ranges, (m, 1))

    return normDataSet, ranges, minVals


# 测试算法

def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('data/datingTestSet2.txt')
    normDataSet, ranges, minVals = autoNorm(datingDataMat)
    m = normDataSet.shape[0]

    numTestVecs = int(m * hoRatio)
    errorCount = 0.0

    for i in range(numTestVecs):
        # 当前的值  样本的多少 对应样本的归类 
        classifierResult = classify0(normDataSet[i, :], normDataSet[numTestVecs:m, :],
                                     datingLabels[numTestVecs: m], 3)
        print("the classifier came back with: %d, the real answer is: %d" %
              (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    rage = errorCount / float(numTestVecs)
    print("错误率: %f" % (rage))


datingClassTest()

# datingDataMat,datingLabels = file2matrix('data/datingTestSet2.txt')

# fig = plt.figure()
# ax = fig.add_subplot(111)
# # 输出第二列和第三列
# ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
# plt.show()
