# Kaggel比赛 : [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)
通过预测在未来两年内某人将经历财务困境的可能性，改善信用评分的状态。

## Description
银行在市场经济中扮演着至关重要的角色。他们决定谁可以获得融资，以及什么条件，可以做出或破坏投资决策。为了让市场和社会发挥作用，个人和企业需要获得信贷。
信用评分算法，对违约概率进行猜测，是银行用来决定是否应该发放贷款的方法。这一竞赛要求参与者通过预测未来两年某人将经历财务困境的可能性，来改善信用评分的状态。
这种竞争的目标是建立一个模型，让借款人可以用来帮助做出最好的财务决策。

## Evaluation
AUC

## Data Description
Training, Test, Sample Entry and Submission Files are provided. Please check the format of the submission file.

Columns
- SeriousDlqin2yrs
- RevolvingUtilizationOfUnsecuredLines
- age
- NumberOfTime30-59DaysPastDueNotWorse
- DebtRatio
- MonthlyIncome
- NumberOfOpenCreditLinesAndLoans
- NumberOfTimes90DaysLate
- NumberRealEstateLoansOrLines
- NumberOfTime60-89DaysPastDueNotWorse
- NumberOfDependents

![myplot](https://github.com/chenxiangzhen/Give_Me_Some_Credit/blob/master/data_description.png)

## 项目流程

- 数据获取，包括获取存量客户及潜在客户的数据。
- 数据预处理，主要工作包括数据清洗、缺失值处理、异常值处理。
- 探索性数据分析，该步骤主要是获取样本总体的大概情况，描述样本总体情况的指标主要有直方图、箱形图等。
- 变量选择，该步骤主要是通过统计学的方法，筛选出对违约状态影响最显著的指标。
- 模型开发，该步骤主要包括变量分段、变量的WOE（证据权重）变换和逻辑回归估算三部分。
- 模型评估，该步骤主要是评估模型的区分能力、预测能力、稳定性，并形成模型评估报告，得出模型是否可以使用的结论。
- 模型评估，该步骤主要是评估模型的区分能力、预测能力、稳定性，并形成模型评估报告，得出模型是否可以使用的结论。
- 建立评分系统，根据信用评分方法，建立自动信用评分系统。


## 数据预处理

在对数据处理之前，需要对数据的缺失值和异常值情况进行了解。Python内有describe()函数，可以了解数据集的缺失值、均值和中位数等。

```
#载入数据
data = pd.read_csv('cs-training.csv')
#数据集确实和分布情况
data.describe().to_csv('DataDescribe.csv')
```
![myplot](https://github.com/chenxiangzhen/Give_Me_Some_Credit/blob/master/datadescription.png)

从上图可知，变量MonthlyIncome和NumberOfDependents存在缺失，变量MonthlyIncome共有缺失值29731个，NumberOfDependents有3924个缺失值。

### 缺失值处理
- 直接删除含有缺失值的样本。
- 根据样本之间的相似性填补缺失值。
- 根据变量之间的相关关系填补缺失值。

变量MonthlyIncome缺失率比较大，所以我们根据变量之间的相关关系填补缺失值，我们采用随机森林法

```
# 用随机森林对缺失值预测填充函数
def set_missing(df):
    # 把已有的数值型特征取出来
    process_df = df.ix[:, [5, 0, 1, 2, 3, 4, 6, 7, 8, 9]]  # 将待填充的放到第一列
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
```
NumberOfDependents变量缺失值比较少，直接删除，对总体模型不会造成太大影响。对缺失值处理完之后，删除重复项。

```
data = set_missing(data)  # 用随机森林填补比较多的缺失值
data = data.dropna()  # 删除比较少的缺失值
data = data.drop_duplicates()  # 删除重复项
data.to_csv('MissingData.csv', index=False)
```


### 异常值处理
缺失值处理完毕后，我们还需要进行异常值处理。异常值是指明显偏离大多数抽样数据的数值，比如个人客户的年龄为0时，通常认为该值为异常值。找出样本总体中的异常值，通常采用离群值检测的方法。

```
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
```

首先，我们发现变量age中存在0，显然是异常值，直接剔除：

```
data = data[data['age'] > 0]  # 年龄等于0的异常值进行剔除
```

对于变量NumberOfTime30-59DaysPastDueNotWorse、NumberOfTimes90DaysLate、NumberOfTime60-89DaysPastDueNotWorse这三个变量，由箱线图可以看出，均存在异常值，且由unique函数可以得知均存在96、98两个异常值，因此予以剔除。

剔除变量NumberOfTime30-59DaysPastDueNotWorse、NumberOfTimes90DaysLate、NumberOfTime60-89DaysPastDueNotWorse的异常值。另外，数据集中好客户为0，违约客户为1，考虑到正常的理解，能正常履约并支付利息的客户为1，所以我们将其取反。


```
data = data[data['NumberOfTime30-59DaysPastDueNotWorse'] < 90]  # 剔除异常值
data['SeriousDlqin2yrs'] = 1-data['SeriousDlqin2yrs']
```
### 数据切分
为了验证模型的拟合效果，我们需要对数据集进行切分，分成训练集和测试集。

```
Y = data['SeriousDlqin2yrs']
X = data.ix[:, 1:]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# print(Y_train)
train = pd.concat([Y_train, X_train], axis=1)
test = pd.concat([Y_test, X_test], axis=1)
clasTest = test.groupby('SeriousDlqin2yrs')['SeriousDlqin2yrs'].count()
train.to_csv('TrainData.csv', index=False)
test.to_csv('TestData.csv', index=False)
print(train.shape)
print(test.shape)
```

## 探索性分析

## 变量选择

### 分箱处理
变量分箱（binning）是对连续变量离散化（discretization）的一种称呼。信用评分卡开发中一般有常用的等距分段、等深分段、最优分段。最优分箱如下：

```
# 定义自动分箱函数
def mono_bin(Y, X, n=20):
    r = 0
    good = Y.sum()
    bad = Y.count()-good
    while np.abs(r) < 1:
        d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, n)})
        d2 = d1.groupby('Bucket', as_index=True)
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
```

针对不能最优分箱的变量，分箱如下：

```
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
```

### WOE
WoE分析， 是对指标分箱、计算各个档位的WoE值并观察WoE值随指标变化的趋势。其中WoE的数学定义是:

```
woe=ln(goodattribute/badattribute)
```

在进行分析时，我们需要对各指标从小到大排列，并计算出相应分档的WoE值。其中正向指标越大，WoE值越小；反向指标越大，WoE值越大。正向指标的WoE值负斜率越大，反响指标的正斜率越大，则说明指标区分能力好。WoE值趋近于直线，则意味指标判断能力较弱。若正向指标和WoE正相关趋势、反向指标同WoE出现负相关趋势，则说明此指标不符合经济意义，则应当予以去除。

### 相关性分析和IV筛选
接下来，我们会用经过清洗后的数据看一下变量间的相关性。注意，这里的相关性分析只是初步的检查，进一步检查模型的VI（证据权重）作为变量筛选的依据。
相关性图我们通过Python里面的seaborn包，调用heatmap()绘图函数进行绘制，实现代码如下：

```
corr = data.corr()#计算各变量的相关性系数
xticks = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']#x轴标签
yticks = list(corr.index)#y轴标签
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
sns.heatmap(corr, annot=True, cmap='rainbow', ax=ax1, annot_kws={'size': 9, 'weight': 'bold', 'color': 'blue'})#绘制相关性系数热力图
ax1.set_xticklabels(xticks, rotation=0, fontsize=10)
ax1.set_yticklabels(yticks, rotation=0, fontsize=10)
plt.show()

```

由上图可以看出，各变量之间的相关性是非常小的。NumberOfOpenCreditLinesAndLoans和NumberRealEstateLoansOrLines的相关性系数为0.43。
接下来，我进一步计算每个变量的Infomation Value（IV）。IV指标是一般用来确定自变量的预测能力。 其公式为：
IV=sum((goodattribute-badattribute)*ln(goodattribute/badattribute))
通过IV值判断变量预测能力的标准是：

```
< 0.02: unpredictive
0.02 to 0.1: weak
0.1 to 0.3: medium
0.3 to 0.5: strong
> 0.5: suspicious
```


IV的实现放在mono_bin()函数里面。

生成的IV图代码：

```
ivlist=[ivx1,ivx2,ivx3,ivx4,ivx5,ivx6,ivx7,ivx8,ivx9,ivx10]#各变量IV
index=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']#x轴的标签
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(1, 1, 1)
x = np.arange(len(index))+1
ax1.bar(x, ivlist, width=0.4)#生成柱状图
ax1.set_xticks(x)
ax1.set_xticklabels(index, rotation=0, fontsize=12)
ax1.set_ylabel('IV(Information Value)', fontsize=14)
#在柱状图上添加数字标签
for a, b in zip(x, ivlist):
    plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=10)
plt.show()

```
可以看出，DebtRatio、MonthlyIncome、NumberOfOpenCreditLinesAndLoans、NumberRealEstateLoansOrLines和NumberOfDependents变量的IV值明显较低，所以予以删除。

## 模型分析
证据权重（Weight of Evidence,WOE）转换可以将Logistic回归模型转变为标准评分卡格式。引入WOE转换的目的并不是为了提高模型质量，只是一些变量不应该被纳入模型，这或者是因为它们不能增加模型值，或者是因为与其模型相关系数有关的误差较大，其实建立标准信用评分卡也可以不采用WOE转换。这种情况下，Logistic回归模型需要处理更大数量的自变量。尽管这样会增加建模程序的复杂性，但最终得到的评分卡都是一样的。
在建立模型之前，我们需要将筛选后的变量转换为WoE值，便于信用评分。

### WOE转换
我们已经能获取了每个变量的分箱数据和woe数据，只需要根据各变量数据进行替换，实现代码如下：

```
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
```
我们将每个变量都进行替换,并将其保存到WoeData.csv文件中：

```
# 替换成woe
data['RevolvingUtilizationOfUnsecuredLines'] = Series(replace_woe(data['RevolvingUtilizationOfUnsecuredLines'], cutx1, woex1))
data['age'] = Series(replace_woe(data['age'], cutx2, woex2))
data['NumberOfTime30-59DaysPastDueNotWorse'] = Series(replace_woe(data['NumberOfTime30-59DaysPastDueNotWorse'], cutx3, woex3))
data['DebtRatio'] = Series(replace_woe(data['DebtRatio'], cutx4, woex4))
data['MonthlyIncome'] = Series(replace_woe(data['MonthlyIncome'], cutx5, woex5))
data['NumberOfOpenCreditLinesAndLoans'] = Series(replace_woe(data['NumberOfOpenCreditLinesAndLoans'], cutx6, woex6))
data['NumberOfTimes90DaysLate'] = Series(replace_woe(data['NumberOfTimes90DaysLate'], cutx7, woex7))
data['NumberRealEstateLoansOrLines'] = Series(replace_woe(data['NumberRealEstateLoansOrLines'], cutx8, woex8))
data['NumberOfTime60-89DaysPastDueNotWorse'] = Series(replace_woe(data['NumberOfTime60-89DaysPastDueNotWorse'], cutx9, woex9))
data['NumberOfDependents'] = Series(replace_woe(data['NumberOfDependents'], cutx10, woex10))
data.to_csv('WoeData.csv', index=False)

test = pd.read_csv('TestData.csv')
# 替换成woe
test['RevolvingUtilizationOfUnsecuredLines'] = Series(replace_woe(test['RevolvingUtilizationOfUnsecuredLines'], cutx1, woex1))
test['age'] = Series(replace_woe(test['age'], cutx2, woex2))
test['NumberOfTime30-59DaysPastDueNotWorse'] = Series(replace_woe(test['NumberOfTime30-59DaysPastDueNotWorse'], cutx3, woex3))
test['DebtRatio'] = Series(replace_woe(test['DebtRatio'], cutx4, woex4))
test['MonthlyIncome'] = Series(replace_woe(test['MonthlyIncome'], cutx5, woex5))
test['NumberOfOpenCreditLinesAndLoans'] = Series(replace_woe(test['NumberOfOpenCreditLinesAndLoans'], cutx6, woex6))
test['NumberOfTimes90DaysLate'] = Series(replace_woe(test['NumberOfTimes90DaysLate'], cutx7, woex7))
test['NumberRealEstateLoansOrLines'] = Series(replace_woe(test['NumberRealEstateLoansOrLines'], cutx8, woex8))
test['NumberOfTime60-89DaysPastDueNotWorse'] = Series(replace_woe(test['NumberOfTime60-89DaysPastDueNotWorse'], cutx9, woex9))
test['NumberOfDependents'] = Series(replace_woe(test['NumberOfDependents'], cutx10, woex10))
test.to_csv('TestWoeData.csv', index=False)
```
### Logisic模型建立
我们直接调用statsmodels包来实现逻辑回归：

```
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

data = pd.read_csv('WoeData.csv')
Y = data['SeriousDlqin2yrs']
X = data.drop(['SeriousDlqin2yrs', 'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberRealEstateLoansOrLines', 'NumberOfDependents'], axis=1)
X1 = sm.add_constant(X)

logit = sm.Logit(Y, X1)
result = logit.fit()
print(result.params)
```
### 模型检验
到这里，我们的建模部分基本结束了。我们需要验证一下模型的预测能力如何。我们使用在建模开始阶段预留的test数据进行检验。通过ROC曲线和AUC来评估模型的拟合能力。
在Python中，可以利用sklearn.metrics，它能方便比较两个分类器，自动计算ROC和AUC。
实现代码：

```
test = pd.read_csv('TestWoeData.csv')
Y_test = test['SeriousDlqin2yrs']
X_test = test.drop(['SeriousDlqin2yrs', 'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans','NumberRealEstateLoansOrLines', 'NumberOfDependents'], axis=1)
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
```

![myplot](https://github.com/chenxiangzhen/Give_Me_Some_Credit/blob/master/myplot.png)