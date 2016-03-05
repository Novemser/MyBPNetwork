# coding=utf-8
__author__ = '胡淦森'
import numpy
import random
import struct
import math
from PIL import Image, ImageDraw
import os
import urllib


# 从选课网下载m张图片验证码
def downloadCheckcodeFromXuanke(m):
    for i in range(0, m):
        url = 'http://xuanke.tongji.edu.cn/CheckImage'
        print ("Downloading %dth check code." % (i + 1))
        file("CheckCode/%04d.jpg" % i, "wb").write(urllib.urlopen(url).read())


# 生成一个 a <= rand < b 的随机数rand
def rand(a, b):
    return (b - a) * random.random() + a


def sigmoid(inX):
    return 1.0 / (1.0 + math.exp(-inX))


def difsigmoid(inX):
    return sigmoid(inX) * (1.0 - sigmoid(inX))


def loadMNISTimage(fileName, datanum=60000):
    images = open(fileName, 'rb')
    buf = images.read()
    index = 0
    magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index)
    print magic, numImages, numRows, numColumns
    index += struct.calcsize('>IIII')
    if magic != 2051:
        raise Exception
    # 像素点数量
    datasize = int(784 * datanum)
    datablock = ">" + str(datasize) + "B"
    nextmatrix = struct.unpack_from(datablock, buf, index)
    nextmatrix = numpy.array(nextmatrix) / 255.0
    nextmatrix = nextmatrix.reshape(datanum, 1, numRows * numColumns)

    return nextmatrix, numImages


def loadMNISTlabels(fileName, datanum=60000):
    labels = open(fileName, 'rb')
    buf = labels.read()
    index = 0
    magic, numLabels = struct.unpack_from('>II', buf, index)
    print magic, numLabels
    index += struct.calcsize('>II')
    if magic != 2049:
        raise Exception

    datablock = ">" + str(datanum) + "B"
    nextmatrix = struct.unpack_from(datablock, buf, index)
    nextmatrix = numpy.array(nextmatrix)
    return nextmatrix, numLabels


def handleCheckCode(img):
    img = img.convert("RGBA")
    pixdata = img.load()
    draw = ImageDraw.Draw(img)
    # 降噪
    # Up
    for y in xrange(0, 3):
        for x in xrange(0, img.size[0]):
            draw.point((x, y), (255, 255, 255, 255))
    # Down
    for y in xrange(img.size[1] - 3, img.size[1]):
        for x in xrange(0, img.size[0]):
            draw.point((x, y), (255, 255, 255, 255))
    # Left
    for x in xrange(0, 2):
        for y in xrange(0, img.size[1]):
            draw.point((x, y), (255, 255, 255, 255))
    # Right
    for x in range(img.size[0] - 8, img.size[0]):
        for y in range(0, img.size[1]):
            draw.point((x, y), (255, 255, 255, 255))
    # Line 1-3
    xx = [10, 19, 28]
    for x in xx:
        for y in xrange(0, img.size[1]):
            draw.point((x, y), (255, 255, 255, 255))

    xx = [-1, 0, 1, -1, 1, -1, 0, 1]
    yy = [1, 1, 1, 0, 0, -1, -1, -1]

    for y in xrange(1, img.size[1] - 1):
        for x in xrange(1, img.size[0] - 1):
            R = pixdata[x, y][0]
            G = pixdata[x, y][1]
            B = pixdata[x, y][2]
            if R < 100 and R == G and G == B:  # 如果某个点依然是接近于黑色
                # 统计周围8个点的色彩情况
                nearDots = 0
                if x != 0 and y != 0:
                    for m in range(8):
                        L = img.getpixel((x + xx[m], y + yy[m]))
                        LR = L[0]
                        LG = L[1]
                        LB = L[2]
                        if LR > 100 and LR == LG and LG == LB:  # 如果某个点依然是接近于白色
                            nearDots += 1  # 计数器加1
                # print nearDots
                if nearDots >= 7:
                    draw.point((x, y), (255, 255, 255, 255))
    # 放大, 去掉灰色的点
    # img = img.resize((img.size[0] * 10, img.size[1] * 10), Image.NORMAL)
    draw = ImageDraw.Draw(img)

    for y in xrange(1, img.size[1] - 1):
        for x in xrange(1, img.size[0] - 1):
            if img.getpixel((x, y))[0] > 200:
                draw.point((x, y), (255, 255, 255, 255))
            elif img.getpixel((x, y))[0] < 100:
                draw.point((x, y), (0, 0, 0, 255))

    # img.save("result.jpg")
    box1 = (2, 3, 10, 15)
    box2 = (11, 3, 19, 15)
    box3 = (20, 3, 28, 15)
    box4 = (29, 3, 37, 15)
    region1 = img.crop(box1)
    # region1.save("r1.png")

    region2 = img.crop(box2)
    # region2.save("r2.png")

    region3 = img.crop(box3)
    # region3.save("r3.png")

    region4 = img.crop(box4)
    # region4.save("r4.png")

    return region1, region2, region3, region4


class BPNN(object):
    # 输入层节点数、隐藏层结点数量、输出层数量、最大迭代次数、训练图片数量
    def __init__(self, inputNodeNumber, hiddenNodeNumber, outputNodeNumber, maxIteration, trainDataNum):
        # 输入、隐藏、输出三层的维度
        self.inputDi = inputNodeNumber
        self.hiddenDi = hiddenNodeNumber
        self.outputDi = outputNodeNumber

        # 迭代的学习速率
        self.learningRate = 0.5
        self.decayRate = 0.2
        self.eps = 0.01
        self.maxIter = maxIteration
        self.trainDataNum = trainDataNum

        # 三层，每一个结点的a值
        self.aInput = [1.0] * self.inputDi
        self.aHidden = [1.0] * self.hiddenDi
        self.aOutput = [1.0] * self.outputDi

        # 生成权重向量
        self.weightMatrixIn2Hi = numpy.zeros((self.inputDi, self.hiddenDi))
        self.weightMatrixHi2Ou = numpy.zeros((self.hiddenDi, self.outputDi))
        self.B = [0.0, 0.0]
        for x in range(self.inputDi):
            for y in range(self.hiddenDi):
                self.weightMatrixIn2Hi[x, y] = rand(-0.2, 0.2)
        for x in range(self.hiddenDi):
            for y in range(self.outputDi):
                self.weightMatrixHi2Ou[x, y] = rand(-0.2, 0.2)

        # 冲量项
        self.ci = numpy.zeros((self.inputDi, self.hiddenDi))
        self.co = numpy.zeros((self.hiddenDi, self.outputDi))

    def loadtraindata(self, fileName):
        self.traindata, self.TotalnumoftrainData = loadMNISTimage(fileName, self.trainDataNum * 2)
        return

    def loadtrainlabel(self, fileName):
        self.trainlabel, self.TotalnumofTrainLabels = loadMNISTlabels(fileName, self.trainDataNum * 2)
        return

    def forwardPropogation(self, inputs):
        # 输入的激活函数
        for i in range(self.inputDi):
            self.aInput[i] = inputs[0, i]

        # 隐藏层的激活函数
        for j in range(self.hiddenDi):
            sum = 0.0
            for i in range(self.inputDi):
                sum += self.weightMatrixIn2Hi[i][j] * self.aInput[i] + self.B[0]
            self.aHidden[j] = sigmoid(sum)

        # Sparse Encoding
        # 隐藏层结点i的平均活跃度计算
        self.pj = []
        for i in range(self.hiddenDi):
            sum = 0
            for j in range(self.inputDi):
                sum += self.aHidden[i] * inputs[0, j]
            self.pj.append(sum / self.hiddenDi)

        # 输出层的激活函数
        for k in range(self.outputDi):
            sum = 0.0
            for j in range(self.hiddenDi):  # [j][k]而不是[j, k].......居然写错了= =
                sum += self.weightMatrixHi2Ou[j][k] * self.aHidden[j] + self.B[1]
            self.aOutput[k] = sigmoid(sum)

        return self.aOutput

    def backwardPropogation(self, targets):
        # 计算输出层的误差项
        output_deltas = [0.0] * self.outputDi
        outlables = []
        for i in range(self.outputDi):
            if i != targets:
                outlables.append(0)
            else:
                outlables.append(1)
        for k in range(self.outputDi):
            output_deltas[k] = -(outlables[k] - self.aOutput[k]) * difsigmoid(self.aHidden[k])

        # 计算隐藏层的误差项
        # Sparse Encoding
        # 加入惩戒因子
        beta = 0.1
        self.p = 0.05
        hidden_deltas = [0.0] * self.hiddenDi
        for j in range(self.hiddenDi):
            error = 0.0
            for k in range(self.outputDi):
                error += output_deltas[k] * self.weightMatrixHi2Ou[j][k]
            error += beta * ((-self.p / self.pj[j]) + (1 - self.p) / (1 - self.pj[j]))
            hidden_deltas[j] = error * difsigmoid(self.aHidden[j])

        # 更新输出层的权重参数
        # 之后改为引入动量项
        N = 0.5
        M = 0.1
        for j in range(self.hiddenDi):
            # cW = 0.0
            for k in range(self.outputDi):  # 自己改了一下更新参数的方法，效果似乎更好- -
                # cW += output_deltas[k] * self.aHidden[j]
                change = output_deltas[k] * self.aHidden[j]
                self.weightMatrixHi2Ou[j][k] = self.weightMatrixHi2Ou[j][k] - N * change + M * self.co[j][k]
                self.co[j][k] = change
                # self.weightMatrixHi2Ou[j][k] -= self.learningRate * change
                # self.weightMatrixHi2Ou[j][k] -= self.learningRate * ((1.0 / self.hiddenDi) * cW + self.decayRate * self.weightMatrixHi2Ou[j][k])

        # 更新输入项的权重参数
        # 之后改为引入动量项
        for i in range(self.inputDi):
            # cW = 0.0
            for j in range(self.hiddenDi):  # 自己改了一下更新参数的方法
                # cW += hidden_deltas[j] * self.aInput[i]
                change = hidden_deltas[j] * self.aInput[i]
                self.weightMatrixIn2Hi[i][j] = self.weightMatrixIn2Hi[i][j] - N * change + M * self.ci[i][j]
                self.ci[i][j] = change
                # self.weightMatrixIn2Hi[i][j] -= self.learningRate * change
                # self.weightMatrixIn2Hi[i][j] -= self.learningRate * ((1.0 / self.inputDi) * cW + self.decayRate * self.weightMatrixIn2Hi[i][j])

        # 计算误差
        error = 0.0
        for k in range(self.outputDi):
            error += 0.5 * (outlables[k] - self.aOutput[k]) ** 2
        return error

    def trainBPNN(self):
        # 迭代maxIter次
        for iter_idx in range(self.maxIter):
            print "Iteration num: ", iter_idx + 1, "====================================="
            error = 0.0
            for idx in range(0, self.trainDataNum):
                # 进行前馈传导计算，利用前向传导公式，得到 L_2, L_3, 直到输出层 L_nl 的激活值
                self.forwardPropogation(self.traindata[idx])
                error += self.backwardPropogation(self.trainlabel[idx])
                print ("Total error in this iteration:%f" % error)
            if error < self.eps:
                print ("Error less than %f, automatically stop iteration." % self.eps)
                break

    def getTrainAccuracy(self):
        accurateCount = 0
        for idx in range(self.trainDataNum, self.trainDataNum * 2):
            predictList = self.forwardPropogation(self.traindata[idx])
            predict = predictList.index(max(predictList))
            result = self.trainlabel[idx]
            print ("Predict: %d" % predict)
            print ("Result: %d" % result)
            if predict == result:
                accurateCount += 1
        print("Accuracy:%f" % (accurateCount * 1.0 / self.trainDataNum))

    def accuracyFromStartToEnd(self, start, end):
        accurateCount = 0
        for idx in range(start, end):
            predictList = self.forwardPropogation(self.traindata[idx])
            predict = predictList.index(max(predictList))
            result = self.trainlabel[idx]
            print ("Predict: %d" % predict)
            print ("Result: %d" % result)
            if predict == result:
                accurateCount += 1
        print("Accuracy:%f" % (accurateCount * 1.0 / (end - start)))
        return accurateCount * 1.0 / (end - start)

    def loadTrainDataFromXuanke(self, totalNum=22):
        self.xuankeTrainNum = totalNum
        data = []
        for i in range(1, self.xuankeTrainNum + 1):
            image = Image.open('TrainSet/code (' + str(i) + ').png')
            for x in xrange(0, image.size[0]):
                for y in xrange(0, image.size[1]):
                    if image.getpixel((x, y))[0] > 200:
                        data.append(1.0)
                    elif image.getpixel((x, y))[0] < 100:
                        data.append(0.0)
        self.nextmatrix = numpy.array(data)
        self.traindata = self.nextmatrix.reshape(22, 1, 8 * 12)
        self.TotalnumoftrainData = self.xuankeTrainNum
        self.trainDataNum = self.xuankeTrainNum
        self.trainlabel = [9, 2, 4, 1, 2, 6, 8, 1, 0, 5, 3, 4, 0, 1, 7, 7, 5, 6, 3, 8, 2, 9]

    def regHandler(self, region, data):
        for x in xrange(0, region.size[0]):
            for y in xrange(0, region.size[1]):
                if region.getpixel((x, y))[0] > 200:
                    data.append(1.0)
                elif region.getpixel((x, y))[0] < 100:
                    data.append(0.0)
        return region, data

    def ocrTheCode(self, checkCodeNum):
        for i in range(0, checkCodeNum):
            data = []
            image = Image.open('CheckCode/%04d.jpg' % i)
            reg1, reg2, reg3, reg4 = handleCheckCode(image)
            reg1, data = self.regHandler(reg1, data)
            reg2, data = self.regHandler(reg2, data)
            reg3, data = self.regHandler(reg3, data)
            reg4, data = self.regHandler(reg4, data)
            # 对四个区域进行识别
            self.nextmatrix = numpy.array(data)
            self.traindata = self.nextmatrix.reshape(4, 1, 8 * 12)
            name = ""
            for idx in range(0, 4):
                predictList = self.forwardPropogation(self.traindata[idx])
                predict = predictList.index(max(predictList))
                name += str(predict)
            print ("Predict of the %dth checkcode: %s" % (i + 1, name))
            # 以识别结果重命名，保存
            os.rename('CheckCode/%04d.jpg' % i, 'Results/' + name + ' ID%d' % i + '.jpg')


if __name__ == '__main__':
    # =================================================================================================================
    print '===============选课网验证码识别部分==============='
    # # 选课网验证码识别部分
    # trainSetNum = 22
    # downloadCheckCodeNum = 100
    # downloadCheckcodeFromXuanke(downloadCheckCodeNum)
    # # 输入层节点数、隐藏层结点数量、输出层数量、最大迭代次数、训练图片数量
    # bpNN_XUANKE = BPNN(8 * 12, 256, 10, 10, trainSetNum)
    # bpNN_XUANKE.loadTrainDataFromXuanke(trainSetNum)
    # bpNN_XUANKE.trainBPNN()
    # accuTrain = bpNN_XUANKE.accuracyFromStartToEnd(0, bpNN_XUANKE.trainDataNum)
    # print ("Accuracy of training set:%f" % accuTrain)
    # # 将下载好的downloadCheckCodeNum个验证码识别之后重命名
    # bpNN_XUANKE.ocrTheCode(downloadCheckCodeNum)
    # downloadCheckcodeFromXuanke(downloadCheckCodeNum)
    # print '==============================================='
    # =================================================================================================================
    # =================================================================================================================
    print '===============MINIST手写数据库识别部分==============='
    # MINIST手写数据库识别部分
    # 输入层节点数、隐藏层结点数量、输出层数量、最大迭代次数、训练图片数量
    bpNN_MINIST = BPNN(28 * 28, 256, 10, 20, 40)
    bpNN_MINIST.loadtraindata('train-images.idx3-ubyte')
    bpNN_MINIST.loadtrainlabel('train-labels.idx1-ubyte')
    bpNN_MINIST.trainBPNN()
    accuTrain = bpNN_MINIST.accuracyFromStartToEnd(0, bpNN_MINIST.trainDataNum)
    acctNext = bpNN_MINIST.accuracyFromStartToEnd(bpNN_MINIST.trainDataNum, bpNN_MINIST.trainDataNum * 2)
    print ("Accuracy of training set:%f" % accuTrain)
    print ("Accuracy of next set:%f" % acctNext)
    print '==============================================='
