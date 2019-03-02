
#数据处理
import numpy as np
import pandas as pd
#绘图
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False

#各种模型、数据处理方法
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
combine_df = pd.concat([train,test])
#
# print(train.Sex.value_counts())
# print('********************************')
# print (train.groupby('Sex')['Survived'].mean())
# print ('***********Train*************')
# print ('test')
# print (train.isnull().sum())
# print ('***********test*************')
# print (test.isnull().sum())


#不考虑Name Ticket
#del train['Name'],test['Name']

#考虑Name
train.groupby(train.Name.apply(lambda x:len(x)))['Survived'].mean().plot()
#plt.show()
#name越长 获救概率越高 因此把名字长度加入特征
combine_df['Name_Len'] = combine_df['Name'].apply(lambda x:len(x))
combine_df['Name_Len'] = pd.qcut(combine_df['Name_Len'],5)

combine_df.groupby(combine_df['Name'].apply(lambda x: x.split(', ')[1]).apply(lambda x: x.split('.')[0]))['Survived'].mean().plot()
#plt.show()
#名字头衔
combine_df['Title'] = combine_df['Name'].apply(lambda x: x.split(', ')[1]).apply(lambda x: x.split('.')[0])
combine_df['Title'] = combine_df['Title'].replace(['Don','Dona', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col','Sir','Dr'],'Mr')
combine_df['Title'] = combine_df['Title'].replace(['Mlle','Ms'], 'Miss')
combine_df['Title'] = combine_df['Title'].replace(['the Countess','Mme','Lady','Dr'], 'Mrs')
df = pd.get_dummies(combine_df['Title'],prefix='Title')
combine_df = pd.concat([combine_df,df],axis=1)
#有女性死亡的家庭 & 有男性存活的家庭
combine_df['Fname'] = combine_df['Name'].apply(lambda x:x.split(',')[0])
combine_df['Familysize'] = combine_df['SibSp']+combine_df['Parch']
dead_female_Fname = list(set(combine_df[(combine_df.Sex=='female') & (combine_df.Age>=12)
                              & (combine_df.Survived==0) & (combine_df.Familysize>1)]['Fname'].values))
survive_male_Fname = list(set(combine_df[(combine_df.Sex=='male') & (combine_df.Age>=12)
                              & (combine_df.Survived==1) & (combine_df.Familysize>1)]['Fname'].values))
combine_df['Dead_female_family'] = np.where(combine_df['Fname'].isin(dead_female_Fname),1,0)
combine_df['Survive_male_family'] = np.where(combine_df['Fname'].isin(survive_male_Fname),1,0)
combine_df = combine_df.drop(['Name','Fname'],axis=1)
#Age
#采用和第一期同样的方法，添加一个小孩子标签

group = combine_df.groupby(['Title', 'Pclass'])['Age']
combine_df['Age'] = group.transform(lambda x: x.fillna(x.median()))
combine_df = combine_df.drop('Title',axis=1)

combine_df['IsChild'] = np.where(combine_df['Age']<=12,1,0)
combine_df['Age'] = pd.cut(combine_df['Age'],5)
combine_df = combine_df.drop('Age',axis=1)
#Familysize
#Familysize再离散化

combine_df['Familysize'] = np.where(combine_df['Familysize']==0, 'solo',
                                    np.where(combine_df['Familysize']<=3, 'normal', 'big'))
df = pd.get_dummies(combine_df['Familysize'],prefix='Familysize')
combine_df = pd.concat([combine_df,df],axis=1).drop(['SibSp','Parch','Familysize'])
#Ticket
combine_df['Ticket_Lett'] = combine_df['Ticket'].apply(lambda x: str(x)[0])
combine_df['Ticket_Lett'] = combine_df['Ticket_Lett'].apply(lambda x: str(x))

combine_df['High_Survival_Ticket'] = np.where(combine_df['Ticket_Lett'].isin(['1', '2', 'P']),1,0)
combine_df['Low_Survival_Ticket'] = np.where(combine_df['Ticket_Lett'].isin(['A','W','3','7']),1,0)
combine_df = combine_df.drop(['Ticket','Ticket_Lett'],axis=1)
#Embarked
#缺省的Embarked用S填充

combine_df.Embarked = combine_df.Embarked.fillna('S')
df = pd.get_dummies(combine_df['Embarked'],prefix='Embarked')
combine_df = pd.concat([combine_df,df],axis=1).drop('Embarked',axis=1)
#Cabin
combine_df['Cabin_isNull'] = np.where(combine_df['Cabin'].isnull(),0,1)
combine_df = combine_df.drop('Cabin',axis=1)
#Pclass
df = pd.get_dummies(combine_df['Pclass'],prefix='Pclass')
combine_df = pd.concat([combine_df,df],axis=1).drop('Pclass',axis=1)
#Sex
df = pd.get_dummies(combine_df['Sex'],prefix='Sex')
combine_df = pd.concat([combine_df,df],axis=1).drop('Sex',axis=1)
#Fare
#缺省值用众数填充，之后进行离散化
combine_df['Fare'] = pd.qcut(combine_df.Fare,3)
df = pd.get_dummies(combine_df.Fare,prefix='Fare').drop('Fare_(-0.001, 8.662]',axis=1)
combine_df = pd.concat([combine_df,df],axis=1).drop('Fare',axis=1)
combine_df.rename(columns={'Fare_(8.662, 26.0]':'Fare_low','Fare_(26.0, 512.329]':'Fare_high'},inplace=True)
#所有特征转化成数值型编码
features = combine_df.drop(["PassengerId","Survived"], axis=1).columns
le = LabelEncoder()
for feature in features:
    le = le.fit(combine_df[feature])
    combine_df[feature] = le.transform(combine_df[feature])
#得到训练/测试数据
X_all = combine_df.iloc[:891,:].drop(["PassengerId","Survived"], axis=1)
Y_all = combine_df.iloc[:891,:]["Survived"]
X_test = combine_df.iloc[891:,:].drop(["PassengerId","Survived"], axis=1)
X_all.to_excel('1.xls')
#模型调优

#分别考察逻辑回归、支持向量机、最近邻、决策树、随机森林、gbdt、xgbGBDT几类算法的性能。
lr = LogisticRegression()
svc = SVC()
knn = KNeighborsClassifier(n_neighbors = 3)
dt = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=300,min_samples_leaf=4,class_weight={0:0.745,1:0.255})
gbdt = GradientBoostingClassifier(n_estimators=500,learning_rate=0.03,max_depth=3)
xgbGBDT = XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
clfs = [lr, svc, knn, dt, rf, gbdt, xgbGBDT]

kfold = 10
cv_results = []
for classifier in clfs :
    cv_results.append(cross_val_score(classifier, X_all, y = Y_all, scoring = "accuracy", cv = kfold))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,
                       "Algorithm":["LR","SVC",'KNN','decision_tree',"random_forest","GBDT","xgbGBDT"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")


class Ensemble(object):
    def __init__(self, estimators):
        self.estimator_names = []
        self.estimators = []
        for i in estimators:
            self.estimator_names.append(i[0])
            self.estimators.append(i[1])
        self.clf = LogisticRegression()

    def fit(self, train_x, train_y):
        for i in self.estimators:
            i.fit(train_x, train_y)
        x = np.array([i.predict(train_x) for i in self.estimators]).T
        y = train_y
        self.clf.fit(x, y)

    def predict(self, x):
        x = np.array([i.predict(x) for i in self.estimators]).T
        # print(x)
        return self.clf.predict(x)

    def score(self, x, y):
        s = precision_score(y, self.predict(x))
        return s

bag = Ensemble([('xgb',xgbGBDT),('lr',lr),('rf',rf),('svc',svc),('gbdt',gbdt)])

score = 0
for i in range(0,10):
    num_test = 0.20
    X_train, X_cv, Y_train, Y_cv = train_test_split(X_all, Y_all, test_size=num_test)
    bag.fit(X_train, Y_train)
    #Y_test = bag.predict(X_test)
    predictions = bag.predict(X_test)
    result = pd.DataFrame({'PassengerId':test['PassengerId'].as_matrix(),
                       'Survived':predictions.astype(np.int32)})
    result.to_csv('data/submission.csv',index=False)
    acc_xgb = round(bag.score(X_cv, Y_cv) * 100, 2)
    score+=acc_xgb


print(score/10)
