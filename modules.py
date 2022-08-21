"""
*****个人数学建模备用代码*****

dependencies：math, sys, scipy, numpy, itertools, matplotlib, statistics, copy, panda
内含类：
AHP - 层次分析法（含经典层次分析法与自适应层次分析法）
FCE - 模糊综合评价方法
MiniTreep - prim算法求最小生成树
TOPSIS - 关于TOPSIS的函数们
内含函数：
intlinp - 整数规划之分支定界法
graycnct - 灰色关联分析
graypredict - 灰色预测模型
relativeFCE - 相对偏差模糊矩阵评价法
relasupFCE - 相对优属度模糊矩阵法
Ftest_pvalue - F检验
shortDistance - Floyd-Warshall算法计算任意两点间最短路径
banch - 分支界限法计算起始节点到其他所有节点的最短距离（最优解）
"""

import math
import sys
from scipy import optimize
from scipy import stats
import numpy as np
import itertools
import matplotlib as mp
import pandas as pd
import statistics as ss
from copy import deepcopy


def intlinp(c, A_ub, b_ub, A_eq=None, b_eq=None, n=0, t=1.0E-9):
    """
        整数规划之分支定界法（求最小代价）。
        （适用于混合整数规划，此时前m(>0)个变量受整数约束，后n(>=0)个变量不受整数约束。）
        其中，c为价值向量，A_ub、b_ub为不等约束条件，A_eq、b_eq为等式约束条件，
        n为不受整数约束的变量的个数，t用于判断解是否为整数（t太小会漏整数，太大会漏小数）。
        返回类型为元组（bestVal，bestX）。
        Warning:混合整数规划代码可行性未经证明，谨慎使用。
    """
    res = optimize.linprog(c, A_ub, b_ub, A_eq, b_eq)  # 计算松弛问题
    if n == len(c):
        return res.fun, res.x  # 非整数规划，直接输出
    else:  # 整数规划
        if type(res.x) is float:
            bestX = [sys.maxsize] * len(c)  # 松弛问题的解是浮点数则bestX取无穷大
        else:
            bestX = res.x  # 解是整数则bestX取解的值
        bestVal = sum([x * y for x, y in zip(c, bestX)])  # 根据bestX求出bestVallue
        ibestX = [x for i, x in enumerate(bestX) if (i < len(bestX) - n)]
        if all((x - math.floor(x)) < t or (math.ceil(x) - x < t) for x in ibestX):
            return bestVal, bestX  # 解都满足整数条件则直接输出
        else:  # 解不满足整数条件则开始分支
            ind = [i for i, x in enumerate(ibestX)
                   if ((x - math.floor(x)) > t and (math.ceil(x) - x > t))][0]  # 第一个非整数的位置
            newcon1 = [0] * len(A_ub[0])
            newcon2 = [0] * len(A_ub[0])
            newcon1[ind] = -1
            newcon2[ind] = 1
            newA1 = A_ub.copy()
            newA2 = A_ub.copy()
            newA1.append(newcon1)
            newA2.append(newcon2)
            newB1 = b_ub.copy()
            newB2 = b_ub.copy()
            newB1.append(-math.ceil(bestX[ind]))
            newB2.append(-math.ceil(bestX[ind]))  # 构造分支问题
            r1 = intlinp(c, newA1, newB1, A_eq, b_eq)
            r2 = intlinp(c, newA1, newB1, A_eq, b_eq)  # 迭代至无可用解
            if r1[0] < r2[0]:
                return r1
            else:
                return r2  # 输出最优解


class AHP:
    """
    ****************
    Created on Tue Jan 26 10:12:30 2021
    自适应层数的层次分析法求权值
    @author: lw
    ****************
    注意：python中list与array运算不一样，严格按照格式输入！
    本层次分析法每个判断矩阵不得超过9阶，各判断矩阵必须是正互反矩阵。
    FA_mx：下一层对上一层的判断矩阵集（包含多个三维数组，默认从目标层向方案层依次输入判断矩阵。同层的判断矩阵按顺序排列，且上层指标不共用下层指标）
    string：默认为'norm'（经典的层次分析法，需输入9标度判断矩阵），若为'auto'（自调节层次分析法，需输入3标度判断矩阵）
    ****************
    示例：
    # 层次分析法的经典9标度矩阵
    goal=[]             #第一层的全部判断矩阵
    goal.append(np.array([[1, 3],
                [1/3 ,1]]))
    criteria1 = np.array([[1, 3],
                          [1/3,1]])
    criteria2=np.array([[1, 1,3],
                        [1,1,3],
                        [1/3,1/3,1]])
    c_all=[criteria1,criteria2]   #第二层的全部判断矩阵
    sample1 = np.array([[1, 1], [1, 1]])
    sample2 = np.array([[1,1,1/3], [1,1,1/3],[3,3,1]])
    sample3 = np.array([[1, 1/3], [3, 1]])
    sample4 = np.array([[1,3,1], [1 / 3, 1, 1/3], [1,3, 1]])
    sample5=np.array([[1,3],[1/3 ,1]])
    sample_all=[sample1,sample2,sample3,sample4,sample5]  #第三层的全部判断矩阵
    FA_mx=[goal,c_all,sample_all]
    A1=AHP(FA_mx)     #经典层次分析法
    A1.run()
    a=A1.CR           #层次单排序的一致性比例（从下往上）
    b=A1.w            #层次单排序的权值（从下往上）
    c=A1.CR_all       #层次总排序的一致性比例（从上往下）
    d=A1.w_all        #层次总排序的权值（从上往下）
    e=sum(d[len(d)-1],[])       #底层指标对目标层的权值
    #可视化
    plt.rcParams['font.sans-serif'] = ['SimHei']  # plt为matplpotlib.pyplot
    plt.rcParams['axes.unicode_minus'] = False
    name=['D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12']
    plt.figure()
    plt.bar(name,e)
    for i,j in enumerate(e):
        plt.text(i,j+0.005,'%.4f'%(np.abs(j)),ha='center',va='top')
    plt.title('底层指标对A的权值')
    plt.show()

    # 自调节层次分析法的3标度矩阵(求在线体系的权值)
    goal = []  # 第一层的全部判断矩阵
    goal.append(np.array([[0, 1],
                          [-1, 0]]))
    criteria1 = np.array([[0, 1],
                          [-1, 0]])
    criteria2 = np.array([[0, 0, 1],
                          [0, 0, 1],
                          [-1, -1, 0]])
    c_all = [criteria1, criteria2]  # 第二层的全部判断矩阵
    sample1 = np.array([[0, 0], [0, 0]])
    sample2 = np.array([[0, 0, -1], [0, 0, -1], [1, 1, 0]])
    sample3 = np.array([[0, -1], [1, 0]])
    sample4 = np.array([[0, 1, 0], [-1, 0, -1], [0, 1, 0]])
    sample5 = np.array([[0, 1], [-1, 0]])
    sample_all = [sample1, sample2, sample3, sample4, sample5]  # 第三层的全部判断矩阵
    FA_mx = [goal, c_all, sample_all]
    A1 = AHP(FA_mx, 'auto')  # 经典层次分析法
    A1.run()
    a = A1.CR  # 层次单排序的一致性比例（从下往上）
    b = A1.w  # 层次单排序的权值（从下往上）
    c = A1.CR_all  # 层次总排序的一致性比例（从上往下）
    d = A1.w_all  # 层次总排序的权值（从上往下）
    e = sum(d[len(d) - 1], [])  # 底层指标对目标层的权值
    #可视化
    plt.rcParams['font.sans-serif'] = ['SimHei']  # plt为matplpotlib.pyplot
    plt.rcParams['axes.unicode_minus'] = False
    name=['D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12']
    plt.figure()
    plt.bar(name,e)
    for i,j in enumerate(e):
        plt.text(i,j+0.005,'%.4f'%(np.abs(j)),ha='center',va='top')
    plt.title('底层指标对A的权值')
    plt.show()
    ****************
    """

    # 初始化函数
    def __init__(self, FA_mx, string='norm'):
        self.RI = np.array([0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51])  # 平均随机一致性指标
        if string == 'norm':
            self.FA_mx = FA_mx  # 所有层级的判断矩阵
        elif string == 'auto':
            self.FA_mx = []
            for i in range(len(FA_mx)):
                temp = []
                for j in range(len(FA_mx[i])):
                    temp.append(self.preprocess(FA_mx[i][j]))
                self.FA_mx.append(temp)  # 自调节层次分析法预处理后的所有层级的判断矩阵
        self.layer_num = len(FA_mx)  # 层级数目
        self.w = []  # 所有层级的权值向量
        self.CR = []  # 所有层级的单排序一致性比例
        self.CI = []  # 所有层级下每个矩阵的一致性指标
        self.RI_all = []  # 所有层级下每个矩阵的平均随机一致性指标
        self.CR_all = []  # 所有层级的总排序一致性比例
        self.w_all = []  # 所有层级指标对目标的权值

    # 输入单个矩阵算权值并一致性检验(特征根法精确求解)
    def count_w(self, mx):
        n = mx.shape[0]
        eig_value, eigen_vectors = np.linalg.eig(mx)
        maxeig = np.max(eig_value)  # 最大特征值
        maxindex = np.argmax(eig_value)  # 最大特征值对应的特征向量
        eig_w = eigen_vectors[:, maxindex] / sum(eigen_vectors[:, maxindex])  # 权值向量
        CI = (maxeig - n) / (n - 1)
        RI = self.RI[n - 1]
        if (n <= 2 and CI == 0):
            CR = 0.0
        else:
            CR = CI / RI
        if (CR < 0.1):
            return CI, RI, CR, list(eig_w.T)
        else:
            print('该%d阶矩阵一致性检验不通过,CR为%.3f' % (n, CR))
            return -1.0, -1.0, -1.0, -1.0

    # 计算单层的所有权值与CR
    def onelayer_up(self, onelayer_mx, index):
        num = len(onelayer_mx)  # 该层矩阵个数
        CI_temp = []
        RI_temp = []
        CR_temp = []
        w_temp = []
        for i in range(num):
            CI, RI, CR, eig_w = self.count_w(onelayer_mx[i])
            if (CR > 0.1):
                print('第%d层的第%d个矩阵未通过一致性检验' % (index, i + 1))
                return
            CI_temp.append(CI)
            RI_temp.append(RI)
            CR_temp.append(CR)
            w_temp.append(eig_w)
        self.CI.append(CI_temp)
        self.RI_all.append(RI_temp)
        self.CR.append(CR_temp)
        self.w.append(w_temp)

    # 计算单层的总排序及该层总的一致性比例
    def alllayer_down(self):
        self.CR_all.append(self.CR[self.layer_num - 1])
        self.w_all.append(self.w[self.layer_num - 1])
        temp = None
        for i in range(self.layer_num - 2, -1, -1):
            if (i == self.layer_num - 2):
                temp = sum(self.w[self.layer_num - 1], [])  # 列表降维，扁平化处理，取上一层的权值向量
            CR_temp = []
            w_temp = []
            CR = sum(np.array(self.CI[i]) * np.array(temp)) / sum(np.array(self.RI_all[i]) * np.array(temp))
            if (CR > 0.1):
                print('第%d层的总排序未通过一致性检验' % (self.layer_num - i))
                return
            for j in range(len(self.w[i])):
                shu = temp[j]
                w_temp.append(list(shu * np.array(self.w[i][j])))
            temp = sum(w_temp, [])  # 列表降维，扁平化处理，取上一层的总排序权值向量
            CR_temp.append(CR)
            self.CR_all.append(CR_temp)
            self.w_all.append(w_temp)
        return

    # 计算所有层的权值与CR,层次总排序
    def run(self):
        for i in range(self.layer_num, 0, -1):
            self.onelayer_up(self.FA_mx[i - 1], i)
        self.alllayer_down()
        return

    # 自调节层次分析法的矩阵预处理过程
    def preprocess(self, mx):
        temp = np.array(mx)
        n = temp.shape[0]
        for i in range(n - 1):
            H = [j for j, x in enumerate(temp[i]) if j > i and x == -1]
            M = [j for j, x in enumerate(temp[i]) if j > i and x == 0]
            L = [j for j, x in enumerate(temp[i]) if j > i and x == 1]
            DL = sum([[i for i in itertools.product(H, M)], [i for i in itertools.product(H, L)],
                      [i for i in itertools.product(M, L)]], [])
            DM = [i for i in itertools.product(M, M)]
            DH = sum([[i for i in itertools.product(L, H)], [i for i in itertools.product(M, H)],
                      [i for i in itertools.product(L, M)]], [])
            if DL:
                for j in DL:
                    if (j[0] < j[1] and i < j[0]):
                        temp[int(j[0])][int(j[1])] = 1
            if DM:
                for j in DM:
                    if (j[0] < j[1] and i < j[0]):
                        temp[int(j[0])][int(j[1])] = 0
            if DH:
                for j in DH:
                    if (j[0] < j[1] and i < j[0]):
                        temp[int(j[0])][int(j[1])] = -1
        for i in range(n):
            for j in range(i + 1, n):
                temp[j][i] = -temp[i][j]
        A = []
        for i in range(n):
            atemp = []
            for j in range(n):
                a0 = 0
                for k in range(n):
                    a0 += temp[i][k] + temp[k][j]
                atemp.append(np.exp(a0 / n))
            A.append(atemp)
        return np.array(A)


def graycnct(std, ce, p=0.5):
    """
    灰色关联分析。
    std为参考数列，ce为比较数列组，p为分辨系数（默认0.5）。
    返回元组（sgm， r） （关联系数，关联度）。
    ***************
    示例：
    std = [9, 9, 9, 9, 9, 9, 9]
    ce = [[8, 9, 8, 7, 5, 2, 9], [7, 8, 7, 5, 7, 3, 8], [9, 7, 9, 6, 6, 4, 7],
        [6, 8, 8, 8, 4, 3, 6], [8, 6, 6, 9, 8, 3, 8],[8, 9, 5, 7, 6, 4, 8]]
    print(mds.graycnct(std, ce))
    ***************
    """
    l = len(std)  # 评价维度数
    n = len(ce)  # 待比较数列的个数
    mini = sys.maxsize
    for i in range(n):
        for k in range(l):
            if abs(std[k] - ce[i][k]) < mini:
                mini = abs(std[k] - ce[i][k])
    maxi = 0
    for i in range(n):
        for k in range(l):
            if abs(std[k] - ce[i][k]) > maxi:
                maxi = abs(std[k] - ce[i][k])
    sgm = []
    for i in range(n):
        sgmi = []
        for k in range(l):
            sgmi.append(
                (mini + p * maxi) / (abs(std[k] - ce[i][k]) + p * maxi)
            )
        sgm.append(sgmi)  # 关联系数矩阵
    r = []
    for i in range(n):
        sumi = 0
        for k in range(l):
            sumi += sgm[i][k]
        r.append(sumi / l)  # 关联度
    return sgm, r


def graypredict(orin, m):
    """
    灰色预测模型（GM(1,1)），同时使用后验差检验、相对偏差检验、级比偏差检验判断灰色预测模型的可行性。
    orin为原始数列,m为向后预测的数据个数。
    返回元组(长度为m的数组（预测出的数据），相对残差检验结果，级比偏差检验结果（0-一般要求，1-较高要求））。
    """
    n = len(orin)
    X0 = np.array(orin)

    # 累加生成
    history_data_agg = [sum(orin[0:i + 1]) for i in range(n)]
    X1 = np.array(history_data_agg)

    # 计算数据矩阵B和数据向量Y
    B = np.zeros([n - 1, 2])
    Y = np.zeros([n - 1, 1])
    for i in range(0, n - 1):
        B[i][0] = -0.5 * (X1[i] + X1[i + 1])
        B[i][1] = 1
        Y[i][0] = X0[i + 1]

    # 计算GM(1,1)微分方程的参数a和u
    # A = np.zeros([2,1])
    A = np.linalg.inv(B.T.dot(B)).dot(B.T).dot(Y)
    a = A[0][0]
    u = A[1][0]

    # 建立灰色预测模型
    XX0 = np.zeros(n)
    XX0[0] = X0[0]
    for i in range(1, n):
        XX0[i] = (X0[0] - u / a) * (1 - math.exp(a)) * math.exp(-a * (i))

    # 模型精度的后验差检验
    e = 0  # 求残差平均值
    for i in range(0, n):
        e += (X0[i] - XX0[i])
    e /= n

    # 求历史数据平均值
    aver = 0
    for i in range(0, n):
        aver += X0[i]
    aver /= n

    # 求历史数据方差
    s12 = 0
    for i in range(0, n):
        s12 += (X0[i] - aver) ** 2
    s12 /= n

    # 求残差方差
    s22 = 0
    for i in range(0, n):
        s22 += ((X0[i] - XX0[i]) - e) ** 2
    s22 /= n

    # 求后验差比值
    C = s22 / s12

    # 求小误差概率
    cout = 0
    for i in range(0, n):
        if abs((X0[i] - XX0[i]) - e) < 0.6754 * math.sqrt(s12):
            cout = cout + 1
        else:
            cout = cout
    P = cout / n

    # 相对残差检验
    eef = -1
    for i in range(0, n):
        ee = abs((X0[i] - XX0[i]) / X0[i])
        if ee < 0.1:
            eef = 1  # 较高要求
        elif ee < 0.2:
            eef = 0  # 一般要求
        else:
            eef = -1  # 不满足要求

    # 级比偏差检验
    ppf = -1
    for i in range(1, n):
        lmd = X0[i - 1] / X0[i]
        pp = abs(1 - lmd * (1 - 0.5 * a) / (1 + 0.5 * a))
        if pp < 0.1:
            ppf = 1  # 较高要求
        elif pp < 0.2:
            ppf = 0  # 一般要求
        else:
            ppf = -1  # 不满足要求

    if C < 0.35 and P > 0.95 and eef > -1 and ppf > -1:
        # 预测精度为一级
        f = np.zeros(m)
        for i in range(m):
            f[i] = (X0[0] - u / a) * (1 - math.exp(a)) * math.exp(-a * (i + n))
        return f, eef, ppf
    else:
        return '灰色预测法不适用'


class FCE:
    """
    模糊综合评价方法，适用于单级模糊评价。如需进行多级模糊评价，可根据以上一级模糊综合评价的结果，作为二级模糊综合评价的关系矩阵，
    进一步得到二级模糊综合评价结果。
    #结果尚未使用加权平均法进行处理。
    Warning:正确性未经检验！
    """

    def __init__(self, dataSet: np.array, method="CVM" or "AHP", operator=0 or 1 or 2 or 3, W=[]):
        """
        输入：
        dataSet为待分析模糊综合评价矩阵，string代表权重分析方法，默认为‘CVM’（变异系数法），可改为‘AHP’（层次分析法），
        operator确定模糊评价算子，共四种,分别为0：取小取大、1：乘后取大、2：乘后求和、3：取小求和。
        如采用层次分析法，请直接指定权重W。
        """
        self.dataSet = dataSet
        self.method = method
        self.operator = operator
        self.W = W

    def norm(dataSet: np.array):
        """
        矩阵归一化。
        return normDataSet, ranges, minVals.
        """
        minVals = dataSet.min(0)  # 取每一列的最小值
        maxVals = dataSet.max(0)  # 取每一列的最大值
        ranges = maxVals - minVals
        vii = (dataSet - minVals) / (maxVals - minVals)
        w = vii / vii.sum()
        return w, ranges, minVals

    def CVM(dataSet: np.array):
        """变异系数法计算权重"""
        si = np.std(dataSet, 1)  # 求每一行标准差
        xi = abs(np.mean(dataSet, 1))  # 求每一行均值
        vi = si / xi
        vvi = FCE.norm(vi)[0]  # 归一化
        return vvi  # 返回权重

    def prindex(W=np.array, R=np.array, operator=0 or 1 or 2 or 3):
        """
        一级模糊评价指标。

        输入：
        W为权重向量。R为模糊综合评价矩阵。
        operator确定模糊评价算子，共四种（默认为1，大概）：

        ————————————————————————————

        |   算子        |   权重体现  | 综合程度  | 利用R的信息 |      类型       |

        |  0 -> 取小取大 |   不明显   |    弱    |   不充分   | 主因素决定型     |

        |  1 -> 乘后取大 |   明显     |   弱    |   不充分   |  主因素突出型     |

        |  2 -> 乘后求和 |   不明显   |    强    |   较充分   | 主因素加权平均型  |

        |  3 -> 取小求和 |   明显     |   强    |   充分    |  全因素加权平均型  |

        ————————————————————————————

        """
        B = []
        if operator == 0:  # 先取小后取大
            for i in range(len(R[0])):
                bi = []
                for j in range(len(W)):
                    bi.append(min(W[j], R.transpose()[i][j]))  # 取小
                bi = np.array(bi)
                B.append(bi.max(0))  # 取大
        elif operator == 1:  # 先乘后取大
            for i in range(len(R[0])):
                bi = []
                for j in range(len(W)):
                    bi.append((W[j] * R.transpose()[i][j]))  # 乘
                bi = np.array(bi)
                B.append(bi.max(0))  # 取大
        elif operator == 2:  # 乘后求和
            for i in range(len(R[0])):
                bi = []
                for j in range(len(W)):
                    bi.append(W[j] * R.transpose()[i][j])  # 乘
                bi = np.array(bi)
                B.append(np.sum(bi, 0))  # 求和
        elif operator == 3:  # 取小求和
            for i in range(len(R[0])):
                bi = []
                for j in range(len(W)):
                    bi.append(min(W[j], R.transpose()[i][j]))  # 取小
                bi = np.array(bi)
                B.append(np.sum(bi, 0))  # 求和
        return B

    def run(self):
        """return B, R, W"""
        R = FCE.norm(self.dataSet)[0]  # 模糊矩阵归一化
        if self.method == 'CVM':  # 确定权重
            W = FCE.CVM(R)
        elif self.method == 'AHP':
            W = self.W
        B = FCE.prindex(W, R, self.operator)  # 建立模糊评价
        return B, R, W


class MiniTreep(object):
    """
    prim算法求最小生成树
    输入示例：
    mini_tree = MiniTreep(['A', 'B', 'C', 'D', 'E', 'F', 'G'],
                         [[10000, 5, 7, 10000, 10000, 10000, 2], [5, 10000, 10000, 9, 10000, 10000, 3],
                          [7, 10000, 10000, 10000, 8, 10000, 10000], [10000, 9, 10000, 10000, 10000, 4, 10000],
                          [10000, 10000, 8, 10000, 10000, 5, 4], [10000, 10000, 10000, 4, 5, 10000, 6],
                          [2, 3, 10000, 10000, 4, 6, 10000], ])
    mini_tree.create_mini_tree(0)
    """
    def __init__(self, vertex, weight):
        """
        最小生成树
        """
        self.vertex = vertex
        self.weight = weight

    def create_mini_tree(self, start):
        """
        最小生成树
        :param start:
        :return:
        """
        visited = []
        # 标记已访问
        visited.append(start)
        v1, v2 = None, None
        while len(visited) < len(self.vertex):
            min_weight = float('inf')
            for v in visited:
                for i in range(len(self.vertex)):
                    # 边没有被访问过且 权重较小
                    if i not in visited and self.weight[v][i] < min_weight:
                        v1 = v
                        v2 = i
                        min_weight = self.weight[v][i]
            visited.append(v2)
            print('%s -> %s weight = %d' % (self.vertex[v1], self.vertex[v2], self.weight[v1][v2]))


class TOPSIS:
    """
    关于TOPSIS的函数们
    https://zhuanlan.zhihu.com/p/37738503
    示例：
    data = pd.DataFrame(
        {'人均专著': [0.1, 0.2, 0.4, 0.9, 1.2], '生师比': [5, 6, 7, 10, 2], '科研经费': [5000, 6000, 7000, 10000, 400],
         '逾期毕业率': [4.7, 5.6, 6.7, 2.3, 1.8]}, index=['院校' + i for i in list('ABCDE')])

    aws = mds.TOPSIS.topsis(data)
    aws[0].to_excel("E:\\aws.xlsx")
    """

    def dataDirection_1(datas, offset=0):
        """极小型指标正向化"""

        def normalization(data):
            return 1 / (data + offset)

        return list(map(normalization, datas))

    def dataDirection_2(datas, x_min, x_max):
        """中间型指标正向化"""

        def normalization(data):
            if data <= x_min or data >= x_max:
                return 0
            elif data > x_min and data < (x_min + x_max) / 2:
                return 2 * (data - x_min) / (x_max - x_min)
            elif data < x_max and data >= (x_min + x_max) / 2:
                return 2 * (x_max - data) / (x_max - x_min)

        return list(map(normalization, datas))

    def dataDirection_3(datas, x_min, x_max, x_minimum, x_maximum):
        """区间型指标正向化，其中(x_min,x_max)为指标的最佳稳定区间，(x_minimum,x_maximum)为最大容忍区间"""

        def normalization(data):
            if data >= x_min and data <= x_max:
                return 1
            elif data <= x_minimum or data >= x_maximum:
                return 0
            elif data > x_max and data < x_maximum:
                return 1 - (data - x_max) / (x_maximum - x_max)
            elif data < x_min and data > x_minimum:
                return 1 - (x_min - data) / (x_min - x_minimum)

        return list(map(normalization, datas))

    def entropyWeight(data):
        """熵权法"""
        data = np.array(data)
        # 归一化
        P = data / data.sum(axis=0)

        # 计算熵值
        E = np.nansum(-P * np.log(P) / np.log(len(data)), axis=0)

        # 计算权系数
        return (1 - E) / (1 - E).sum()

    def topsis(data, weight=None):
        # 归一化
        data = data / np.sqrt((data ** 2).sum())

        # 最优最劣方案
        Z = pd.DataFrame([data.min(), data.max()], index=['负理想解', '正理想解'])

        # 距离
        weight = TOPSIS.entropyWeight(data) if weight is None else np.array(weight)
        Result = data.copy()
        Result['正理想解'] = np.sqrt(((data - Z.loc['正理想解']) ** 2 * weight).sum(axis=1))
        Result['负理想解'] = np.sqrt(((data - Z.loc['负理想解']) ** 2 * weight).sum(axis=1))

        # 综合得分指数
        Result['综合得分指数'] = Result['负理想解'] / (Result['负理想解'] + Result['正理想解'])
        Result['排序'] = Result.rank(ascending=False)['综合得分指数']

        return Result, Z, weight


def reladvtFCE(A: np.array, U, W):
    """
    相对偏差模糊矩阵评价法。
    A为实际数据，U为理想数据，W为权重。
    :return:F：评价系数
    """
    R = np.zeros(np.shape(A))
    for i in range(len(A)):
        for j in range(len(A[0])):
            R[i][j] = abs(A[i][j] - U[i]) / (A[i].max() - A[i].min())  # 建立相对偏差矩阵
    F = np.zeros(np.shape(A[0]))
    for j in range(len(A[0])):
        for i in range(len(A)):
            F[j] += W[i] * R[i][j]  # 加权平均
    return F


def relasupFCE(A: np.array, R, W):
    """
    相对优属度模糊矩阵法
    （其实就是一个加权平均）
    """
    F = np.zeros(np.shape(A[0]))
    for j in range(len(A[0])):
        for i in range(len(A)):
            F[j] += W[i] * R[i][j]  # 加权平均
    return F


def Ftest_pvalue(d1, d2):
    """F检验"""
    df1 = len(d1) - 1
    df2 = len(d2) - 1
    F = ss.variance(d1) / ss.variance(d2)
    single_tailed_pval = stats.f.cdf(F, df1, df2)
    double_tailed_pval = single_tailed_pval * 2
    return double_tailed_pval


def shortDistance(dis):
    """
    Floyd-Warshall算法是解决任意两点间的最短路径的一种算法。通常可以在任何图中使用，包括有向图、带负权边的图。
    存储方式采用邻接矩阵。
    dis为原图邻接矩阵（方阵）
    输入示例：
        dis = [[0, 1, 2, math.inf, 4],
       [1, 0, math.inf, 8, 2],
       [2, math.inf, 0, math.inf, 6],
       [math.inf, 8, math.inf, 0, 3],
       [4, 2, 6, 3, 0]]
    转载自：https://blog.csdn.net/qq_34950042/article/details/88387797?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-88387797-blog-116187720.pc_relevant_multi_platform_featuressortv2dupreplace&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-88387797-blog-116187720.pc_relevant_multi_platform_featuressortv2dupreplace&utm_relevant_index=1
    """
    node_num = len(dis)
    new_dis = dis
    for i in range(node_num):  # 十字交叉法的位置位置，先列后行
        for j in range(node_num):  # 列 表示dis[j][i]的值，即j->i
            for k in range(j + 1, node_num):  # 行 表示dis[i][k]的值，即i->k，i只是一个桥梁而已
                # 先列后行，形成一个传递关系，若比原来距离小，则更新
                if new_dis[j][k] > new_dis[j][i] + new_dis[i][k]:
                    new_dis[j][k] = new_dis[j][i] + new_dis[i][k]
                    new_dis[k][j] = new_dis[j][i] + new_dis[i][k]
    return new_dis


def banch(graph, start):
    """
    分支界限：计算起始节点到其他所有节点的最短距离（最优解）
    1.将起始节点入队，并且初始化起始节点到其他所有节点距离为inf，用costs
    2.检测起始节点的到子节点的距离是否变短，若是，则将其子节点入队
    3.子节点全部检测完，则将起始节点出队，
    4.让队列中的第一个元素作为新的起始节点，重复1,2,3,4
    5.对队列为空，则退出循环
    输入示例：
    graph = {1: {2: 4, 3: 2, 4: 5},
         2: {5: 7, 6: 5},
         3: {6: 9},
         4: {5: 2, 7: 7},
         5: {8: 4},
         6: {10: 6},
         7: {9: 3},
         8: {10: 7},
         9: {10: 8},
         10: {}
         }

    """
    costs = {}  # 记录start到其他所有点的距离
    trace = {start: [start]}  # 记录start到其他所有点的路径

    # 初始化costs
    for key in graph.keys():
        costs[key] = math.inf
    costs[start] = 0

    queue = [start]  # 初始化queue

    while len(queue) != 0:
        head = queue[0]  # 起始节点
        for key in graph[head].keys():  # 遍历起始节点的子节点
            dis = graph[head][key] + costs[head]
            if costs[key] > dis:
                costs[key] = dis
                temp = deepcopy(trace[head])  # 深拷贝
                temp.append(key)
                trace[key] = temp  # key节点的最优路径为起始节点最优路径+key
                queue.append(key)

        queue.pop(0)  # 删除原来的起始节点
    return costs, trace


