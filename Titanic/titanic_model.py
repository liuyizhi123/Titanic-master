import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False

data_train = pd.read_csv('data/train.csv')

# 乘客各属性分布
fig = plt.figure(figsize=(14,8))
fig.set(alpha=0.2)

plt.subplot2grid((2,3),(0,0))
data_train.Survived.value_counts().plot(kind='bar')
plt.title(u'获救情况（1为获救）')
plt.ylabel(u'人数')

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind='bar')
plt.title(u'乘客等级分布')
plt.ylabel(u'人数')

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived,data_train.Age)
plt.ylabel(u"年龄")
plt.grid(b=True, which='major', axis='y')
plt.title(u"按年龄看获救分布 (1为获救)")

plt.subplot2grid((2,3),(1,0),colspan=2)
data_train.Age[data_train.Pclass ==1].plot(kind='kde')
data_train.Age[data_train.Pclass ==2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"年龄")
plt.ylabel(u"密度")
plt.title(u"各等级的乘客年龄分布")
plt.legend((u'头等舱',u'2等舱',u'3等舱'),loc=0)

plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"各登船口岸上船人数")
plt.ylabel(u"人数")

#plt.show()

#各乘客等级的获救情况
Survived_0 = data_train.Pclass[data_train.Survived==0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived==1].value_counts()
df = pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
#print(df)
df.plot(kind='bar',stacked = True)
plt.title(u"各乘客等级的获救情况")
plt.xlabel(u"乘客等级")
plt.ylabel(u"人数")
#plt.show()

# #各性别的获救情况
# Survived_n = data_train.Sex[data_train.Survived == 0].value_counts()
# Survived_y = data_train.Sex[data_train.Survived == 1].value_counts()
# df1 = pd.DataFrame({u'获救':Survived_y, u'未获救':Survived_n})
# print(df1)
# df1.plot(kind='bar',stacked = True)
# plt.title(u"各乘客性别的获救情况")
# plt.xlabel(u"乘客性别")
# plt.ylabel(u"人数")
# plt.show()

Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df1 = pd.DataFrame({u'男性':Survived_m, u'女性':Survived_f})
df1.plot(kind='bar', stacked=True)
plt.title(u"按性别看获救情况")
plt.xlabel(u"获救情况（1为获救）")
plt.ylabel(u"人数")
#plt.show()

#各种舱级别情况下各性别的获救情况
fig=plt.figure(figsize=(10,5))
fig.set(alpha=0.65)
plt.xticks([])
plt.yticks([])
plt.title(u"根据舱等级和性别的获救情况")
ax1 = fig.add_subplot(141)
data_train.Survived[data_train.Sex=='female'][data_train.Pclass !=3].value_counts().plot(kind='bar',label='female highclass',color='#FA2479')
ax1.set_xticklabels([u"获救", u"未获救"],rotation=0)
ax1.legend([u"女性/高级舱"],loc=0)


ax2 = fig.add_subplot(142,sharey=ax1)
data_train.Survived[data_train.Sex=='female'][data_train.Pclass==3].value_counts().plot(kind='bar',label='female lowclass',color='pink')
ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
ax2.legend([u"女性/低级舱"], loc=0)

ax3=fig.add_subplot(143, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male highclass',color='lightblue')
ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)
ax3.legend([u"男性/高级舱"], loc='best')


ax4=fig.add_subplot(144, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male lowclass', color='steelblue')
ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)
ax4.legend([u"男性/低级舱"], loc='best')
#plt.show()

Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df2 =pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df2.plot(kind='bar', stacked=True)
plt.title(u"各登录港口乘客的获救情况")
plt.xlabel(u"登录港口")
plt.ylabel(u"人数")
#plt.show()

Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df3 = pd.DataFrame({u'有':Survived_cabin, u'无':Survived_nocabin}).transpose()
df3.plot(kind='bar', stacked=True)
plt.title(u"按Cabin有无看获救情况")
plt.xlabel(u"Cabin有无")
plt.ylabel(u"人数")
#plt.show()

# 使用 RandomForestRegressor 填补缺失的年龄属性
from sklearn.ensemble import  RandomForestRegressor
def set_missing_ages(df):

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    # y即目标年龄
    y = known_age[:,0]
    X = known_age[:, 1:]
    #fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
    rfr.fit(X,y)
    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:,1:])
    df.loc[(df.Age.isnull()),'Age'] = predictedAges
    return  df,rfr

def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()),'Cabin'] = 'Yes'
    df.loc[(df.Cabin.isnull()), 'Cabin'] = 'No'
    return df

data_train_pro,rfr=set_missing_ages(data_train)
data_train_processed= set_Cabin_type(data_train_pro)

#print(data_train_processed)
#对类目型的特征因子化

dummis_Cabin = pd.get_dummies(data_train_processed['Cabin'],prefix='Cabin')
dummis_Embarked = pd.get_dummies(data_train_processed['Embarked'],prefix='Embarked')
dummis_Sex = pd.get_dummies(data_train_processed['Sex'],prefix='Sex')
dummis_Pclass = pd.get_dummies(data_train_processed['Pclass'],prefix='Pclass')

df = pd.concat([data_train_processed,dummis_Cabin,dummis_Embarked,dummis_Sex,dummis_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked','Cabin_No'], axis=1, inplace=True)


#数据标准特征化
import  sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
scale_param = scaler.fit(df[['Age','Fare']])
df['Age_scaled'] = scaler.fit_transform(df[['Age','Fare']], scale_param)[:,0]
df['Fare_scaled'] = scaler.fit_transform(df[['Age','Fare']], scale_param)[:,1]

#print(df)

from sklearn.linear_model import  LogisticRegression

# 用正则取出我们要的属性值
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

y = train_np[:,0]
x = train_np[:,1:]

clf = LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
clf.fit(x,y)
#print(clf)
#把模型系数和属性关联分析
coef_dt = pd.DataFrame({'columns':list(train_df.columns)[1:],'coef':list(clf.coef_.T)})
#print(coef_dt)



#对test数据进行预处理
data_test = pd.read_csv('data/test.csv')
data_test.loc[(data_test.Fare.isnull()),'Fare'] = 0
#用RandomForestRegressor模型填年龄缺失值
tmp_df = data_test[['Age','Fare','Parch','SibSp','Pclass']]
null_age = tmp_df[data_test.Age.isnull()]
X = null_age.iloc[:,1:]
predictedAges = rfr.predict(X)
data_test.loc[data_test.Age.isnull(),'Age'] = predictedAges

data_test_processed = set_Cabin_type(data_test)
dummis_Cabin = pd.get_dummies(data_test_processed['Cabin'],prefix='Cabin')
dummis_Embarked = pd.get_dummies(data_test_processed['Embarked'],prefix='Embarked')
dummis_Sex = pd.get_dummies(data_test_processed['Sex'],prefix='Sex')
dummis_Pclass = pd.get_dummies(data_test_processed['Pclass'],prefix='Pclass')

df_test = pd.concat([data_test_processed,dummis_Cabin,dummis_Embarked,dummis_Sex,dummis_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked','Cabin_No'], axis=1, inplace=True)

scaler = preprocessing.StandardScaler()
scale_param_test = scaler.fit(df_test[['Age','Fare']])
df_test['Age_scaled'] = scaler.fit_transform(df_test[['Age','Fare']], scale_param)[:,0]
df_test['Fare_scaled'] = scaler.fit_transform(df_test[['Age','Fare']], scale_param)[:,1]
#print(df_test)

#预测结果
test_X = df_test.filter(regex = 'Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test_X)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(),
                       'Survived':predictions.astype(np.int32)})
result.to_csv('data/logistic_regression_predictions.csv',index=False)