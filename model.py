import pandas as pd
from pandas import Series,DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
import numpy as np
import math
import scipy.stats.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc
from pylab import mpl


# 用随机森林对缺失值预测填充函数
def set_missing(df):
    # 把已有的数值型特征取出来，将待填充的列放到第一列
    process_df = df.ix[:, [5, 0, 1, 2, 3, 4, 6, 7, 8, 9]]
    # 分成已知该特征和未知该特征两部分
    known = process_df[process_df.MonthlyIncome.notnull()].as_matrix()
    unknown = process_df[process_df.MonthlyIncome.isnull()].as_matrix()
    # X为特征属性值
    X = known[:, 1:]
    # y为结果标签值
    y = known[:, 0]
    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=3, n_jobs=-1)
    rfr.fit(X, y)
    # 用得到的模型进行未知特征值预测
    predicted = rfr.predict(unknown[:, 1:]).round(0)
    print(predicted)
    # 用得到的预测结果填补原缺失数据
    df.loc[(df.MonthlyIncome.isnull()), 'MonthlyIncome'] = predicted
    return df


def outlier_processing(df, col):
    """
    离群值处理
    :param df:
    :param col:
    :return:
    """
    s = df[col]
    oneQuoter = s.quantile(0.25)
    threeQuote = s.quantile(0.75)
    irq = threeQuote-oneQuoter
    min = oneQuoter-1.5*irq
    max = threeQuote+1.5*irq
    df = df[df[col] <= max]
    df = df[df[col] >= min]
    return df


# 定义自动分箱函数
def mono_bin(Y, X, n=20):
    r = 0
    good = Y.sum()
    bad = Y.count()-good
    while np.abs(r) < 1:
        pdqcut = pd.qcut(X, n).value_counts()
        # pd.qcut根据这些值的频率来选择箱子的均匀间隔
        d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, n)})
        d2 = d1.groupby('Bucket', as_index=True)
        # 等级相关程度的统计分析指标
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
        n = n - 1
    d3 = pd.DataFrame(d2.X.min(), columns=['min'])
    d3['min'] = d2.min().X
    d3['max'] = d2.max().X
    d3['sum'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    d3['woe'] = np.log((d3['rate']/(1-d3['rate']))/(good/bad))
    d3['goodattribute'] = d3['sum']/good
    d3['badattribute'] = (d3['total']-d3['sum'])/bad
    iv = ((d3['goodattribute']-d3['badattribute'])*d3['woe']).sum()
    d4 = (d3.sort_index(by='min'))
    print("=" * 60)
    print(d4)
    cut = []
    cut.append(float('-inf'))
    for i in range(1, n+1):
        qua = X.quantile(i/(n+1))
        cut.append(round(qua, 4))
    cut.append(float('inf'))
    woe = list(d4['woe'].round(3))
    return d4, iv, cut, woe


# 自定义分箱函数
def self_bin(Y, X, cat):
    good = Y.sum()
    bad = Y.count()-good
    d1 = pd.DataFrame({'X': X, 'Y': Y, 'Bucket': pd.cut(X, cat)})
    d2 = d1.groupby('Bucket', as_index=True)
    d3 = pd.DataFrame(d2.X.min(), columns=['min'])
    d3['min'] = d2.min().X
    d3['max'] = d2.max().X
    d3['sum'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    d3['woe'] = np.log((d3['rate'] / (1 - d3['rate'])) / (good / bad))
    d3['goodattribute'] = d3['sum'] / good
    d3['badattribute'] = (d3['total'] - d3['sum']) / bad
    iv = ((d3['goodattribute'] - d3['badattribute']) * d3['woe']).sum()
    d4 = (d3.sort_index(by='min'))
    print("=" * 60)
    print(d4)
    woe = list(d4['woe'].round(3))
    return d4, iv, woe


# 用woe代替
def replace_woe(series, cut, woe):
    list = []
    i = 0
    while i < len(series):
        value = series[i]
        j = len(cut)-2
        m = len(cut)-2
        while j >= 0:
            if value >= cut[j]:
                j = -1
            else:
                j -= 1
                m -= 1
        list.append(woe[m])
        i += 1
    return list


# 计算分数函数
def get_score(coe,woe,factor):
    scores=[]
    for w in woe:
        score=round(coe*w*factor,0)
        scores.append(score)
    return scores


# 根据变量计算分数
def compute_score(series,cut,score):
    list = []
    i = 0
    while i < len(series):
        value = series[i]
        j = len(cut) - 2
        m = len(cut) - 2
        while j >= 0:
            if value >= cut[j]:
                j = -1
            else:
                j -= 1
                m -= 1
        list.append(score[m])
        i += 1
    return list


if __name__ == '__main__':
    # 载入数据
    data = pd.read_csv('cs-training.csv')
    # 数据集缺失和分布情况
    data.describe().to_csv('DataDescribe.csv')
    # 用随机森林填补比较多的缺失值
    data = set_missing(data)
    # 删除比较少的缺失值
    data = data.dropna()
    # 删除重复项
    data = data.drop_duplicates()
    data.describe().to_csv('MissingDataDescribe.csv')

    # 去除异常值
    # data['age'] = outlier_processing(data, 'age')
    data = data[data['age'] > 0]  # 年龄等于0的异常值进行剔除
    data = data[data['NumberOfTime30-59DaysPastDueNotWorse'] < 90]  # 剔除异常值
    data['SeriousDlqin2yrs'] = 1 - data['SeriousDlqin2yrs']  # 数据集中好客户为0，违约客户为1，考虑到正常的理解，能正常履约并支付利息的客户为1，所以我们将其取反。

    # 划分训练集和测试集
    Y = data['SeriousDlqin2yrs']
    X = data.ix[:, 1:]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    # print(Y_train)
    train = pd.concat([Y_train, X_train], axis=1)
    test = pd.concat([Y_test, X_test], axis=1)
    classTest = test.groupby('SeriousDlqin2yrs')['SeriousDlqin2yrs'].count()
    print('classTest:', classTest)
    train.to_csv('TrainData.csv', index=False)
    test.to_csv('TestData.csv', index=False)
    print(train.shape)
    print(test.shape)

    data = pd.read_csv('TrainData.csv')
    pinf = float('inf')  # 正无穷大
    ninf = float('-inf')  # 负无穷大

    # 最优分箱
    dfx1, ivx1, cutx1, woex1 = mono_bin(data.SeriousDlqin2yrs, data.RevolvingUtilizationOfUnsecuredLines, n=10)
    dfx2, ivx2, cutx2, woex2 = mono_bin(data.SeriousDlqin2yrs, data.age, n=10)
    dfx4, ivx4, cutx4, woex4 = mono_bin(data.SeriousDlqin2yrs, data.DebtRatio, n=20)
    dfx5, ivx5, cutx5, woex5 = mono_bin(data.SeriousDlqin2yrs, data.MonthlyIncome, n=10)

    # 连续变量离散化
    cutx3 = [ninf, 0, 1, 3, 5, pinf]
    cutx6 = [ninf, 1, 2, 3, 5, pinf]
    cutx7 = [ninf, 0, 1, 3, 5, pinf]
    cutx8 = [ninf, 0, 1, 2, 3, pinf]
    cutx9 = [ninf, 0, 1, 3, pinf]
    cutx10 = [ninf, 0, 1, 2, 3, 5, pinf]

    dfx3, ivx3, woex3 = self_bin(data.SeriousDlqin2yrs, data['NumberOfTime30-59DaysPastDueNotWorse'], cutx3)
    dfx6, ivx6, woex6 = self_bin(data.SeriousDlqin2yrs, data['NumberOfOpenCreditLinesAndLoans'], cutx6)
    dfx7, ivx7, woex7 = self_bin(data.SeriousDlqin2yrs, data['NumberOfTimes90DaysLate'], cutx7)
    dfx8, ivx8, woex8 = self_bin(data.SeriousDlqin2yrs, data['NumberRealEstateLoansOrLines'], cutx8)
    dfx9, ivx9, woex9 = self_bin(data.SeriousDlqin2yrs, data['NumberOfTime60-89DaysPastDueNotWorse'], cutx9)
    dfx10, ivx10, woex10 = self_bin(data.SeriousDlqin2yrs, data['NumberOfDependents'], cutx10)

    ivlist = [ivx1, ivx2, ivx3, ivx4, ivx5, ivx6, ivx7, ivx8, ivx9, ivx10]
    index = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(1, 1, 1)
    x = np.arange(len(index)) + 1
    ax1.bar(x, ivlist, width=0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(index, rotation=0, fontsize=12)
    ax1.set_ylabel('IV(Information Value)', fontsize=14)
    for a, b in zip(x, ivlist):
        plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=10)

    # 替换成woe
    data['RevolvingUtilizationOfUnsecuredLines'] = Series(
        replace_woe(data['RevolvingUtilizationOfUnsecuredLines'], cutx1, woex1))
    data['age'] = Series(replace_woe(data['age'], cutx2, woex2))
    data['NumberOfTime30-59DaysPastDueNotWorse'] = Series(
        replace_woe(data['NumberOfTime30-59DaysPastDueNotWorse'], cutx3, woex3))
    data['DebtRatio'] = Series(replace_woe(data['DebtRatio'], cutx4, woex4))
    data['MonthlyIncome'] = Series(replace_woe(data['MonthlyIncome'], cutx5, woex5))
    data['NumberOfOpenCreditLinesAndLoans'] = Series(replace_woe(data['NumberOfOpenCreditLinesAndLoans'], cutx6, woex6))
    data['NumberOfTimes90DaysLate'] = Series(replace_woe(data['NumberOfTimes90DaysLate'], cutx7, woex7))
    data['NumberRealEstateLoansOrLines'] = Series(replace_woe(data['NumberRealEstateLoansOrLines'], cutx8, woex8))
    data['NumberOfTime60-89DaysPastDueNotWorse'] = Series(
        replace_woe(data['NumberOfTime60-89DaysPastDueNotWorse'], cutx9, woex9))
    data['NumberOfDependents'] = Series(replace_woe(data['NumberOfDependents'], cutx10, woex10))
    data.to_csv('WoeData.csv', index=False)

    # 逻辑回归模型
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    data = pd.read_csv('WoeData.csv')
    Y = data['SeriousDlqin2yrs']
    X = data.drop(['SeriousDlqin2yrs',
                   'DebtRatio',
                   'MonthlyIncome',
                   'NumberOfOpenCreditLinesAndLoans',
                   'NumberRealEstateLoansOrLines',
                   'NumberOfDependents'], axis=1)
    X1 = sm.add_constant(X)
    logit = sm.Logit(Y, X1)
    result = logit.fit()
    print(result.params)
    print(result.summary())

    test = pd.read_csv('TestData.csv')
    # 替换成woe
    test['RevolvingUtilizationOfUnsecuredLines'] = Series(
        replace_woe(test['RevolvingUtilizationOfUnsecuredLines'], cutx1, woex1))
    test['age'] = Series(replace_woe(test['age'], cutx2, woex2))
    test['NumberOfTime30-59DaysPastDueNotWorse'] = Series(
        replace_woe(test['NumberOfTime30-59DaysPastDueNotWorse'], cutx3, woex3))
    test['DebtRatio'] = Series(replace_woe(test['DebtRatio'], cutx4, woex4))
    test['MonthlyIncome'] = Series(replace_woe(test['MonthlyIncome'], cutx5, woex5))
    test['NumberOfOpenCreditLinesAndLoans'] = Series(replace_woe(test['NumberOfOpenCreditLinesAndLoans'], cutx6, woex6))
    test['NumberOfTimes90DaysLate'] = Series(replace_woe(test['NumberOfTimes90DaysLate'], cutx7, woex7))
    test['NumberRealEstateLoansOrLines'] = Series(replace_woe(test['NumberRealEstateLoansOrLines'], cutx8, woex8))
    test['NumberOfTime60-89DaysPastDueNotWorse'] = Series(
        replace_woe(test['NumberOfTime60-89DaysPastDueNotWorse'], cutx9, woex9))
    test['NumberOfDependents'] = Series(replace_woe(test['NumberOfDependents'], cutx10, woex10))
    test.to_csv('TestWoeData.csv', index=False)

    test = pd.read_csv('TestWoeData.csv')
    Y_test = test['SeriousDlqin2yrs']
    X_test = test.drop(['SeriousDlqin2yrs', 'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
                        'NumberRealEstateLoansOrLines', 'NumberOfDependents'], axis=1)
    X3 = sm.add_constant(X_test)
    resu = result.predict(X3)
    fpr, tpr, threshold = roc_curve(Y_test, resu)
    rocauc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % rocauc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('真正率')
    plt.xlabel('假正率')
    plt.show()

    # 计算分数
    # coe为逻辑回归模型的系数
    coe = [result.params['const'],
           result.params['RevolvingUtilizationOfUnsecuredLines'],
           result.params['age'],
           result.params['NumberOfTime30-59DaysPastDueNotWorse'],
           result.params['NumberOfTimes90DaysLate'],
           result.params['NumberOfTime60-89DaysPastDueNotWorse']]
    # 我们取600分为基础分值，PDO为20（每高20分好坏比翻一倍），好坏比取20。
    p = 20 / math.log(2)
    q = 600 - 20 * math.log(20) / math.log(2)
    baseScore = round(q + p * coe[0], 0)
    # 各项部分分数
    x1 = get_score(coe[1], woex1, p)
    x2 = get_score(coe[2], woex2, p)
    x3 = get_score(coe[3], woex3, p)
    x7 = get_score(coe[4], woex7, p)
    x9 = get_score(coe[5], woex9, p)
    print(x1, x2, x3, x7, x9)
    test1 = pd.read_csv('TestData.csv')
    test1['BaseScore'] = Series(np.zeros(len(test1))) + baseScore
    test1['x1'] = Series(compute_score(test1['RevolvingUtilizationOfUnsecuredLines'], cutx1, x1))
    test1['x2'] = Series(compute_score(test1['age'], cutx2, x2))
    test1['x3'] = Series(compute_score(test1['NumberOfTime30-59DaysPastDueNotWorse'], cutx3, x3))
    test1['x7'] = Series(compute_score(test1['NumberOfTimes90DaysLate'], cutx7, x7))
    test1['x9'] = Series(compute_score(test1['NumberOfTime60-89DaysPastDueNotWorse'], cutx9, x9))
    test1['Score'] = test1['x1'] + test1['x2'] + test1['x3'] + test1['x7'] + test1['x9'] + baseScore
    test1.to_csv('ScoreData2.csv', index=False)
    plt.show()