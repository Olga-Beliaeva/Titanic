import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

from scipy.stats import uniform
from scipy.stats import randint as sp_randint

from sklearn import metrics
from sklearn import set_config
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import make_column_selector as selector
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler,  StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report,accuracy_score, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV, StratifiedKFold

np.random.seed(0)

lb_make = LabelEncoder()
K_fold  = StratifiedKFold(n_splits=10)

# **I. PREPROCESSING DATA**
titanic_df = pd.read_csv('titanic_train_kagle.csv')
test_df = pd.read_csv('titanic_test_kagle.csv')
titanic_df.head(3)
print(f'train_set shape: {titanic_df.shape}, test_set shape: {test_df.shape}')

#- Survived: Survival (0 = No; 1 = Yes) 
#- Pcass: Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd) 
#- SibSp: Number of Siblings/Spouses Aboard 
#- Parch: Number of Parents/Children Aboard 
#- Fare: Passenger Fare (British pound) 
#- Sex: Sex (male = 1, female = 0) 
#- Embarked: Port of Embarkation (C = Cherbourg = 2; Q = Queenstown = 3; S = Southampton = 1) 
#- Age: Age

#**MISSING DATA**
train_missing = titanic_df.isnull().sum().to_frame().reset_index()
test_missing = test_df.isnull().sum().to_frame().reset_index()
missing_data = train_missing.merge(test_missing, on='index').rename(columns={'0_x':'train', '0_y':'test'})
missing_data
trd, ted = titanic_df.copy(), test_df.copy()

data = [trd, ted]
for dataset in data:
    dataset.drop(['Ticket', 'Cabin'], axis=1, inplace=True)

#**UNIQUE DATA**
train_unique = trd.nunique().to_frame().reset_index()
test_unique = ted.nunique().to_frame().reset_index()
unique_data = train_unique.merge(test_unique, on='index').rename(columns={'0_x':'train', '0_y':'test'})
unique_data

#**QUICK DATA OVERVIEW**
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio # без этого фигуры ниже не отображаются
pio.renderers.default = "iframe"

fig1=px.histogram(titanic_df,
             x='Sex', 
             barmode="group",
             marginal="box", nbins=100,
             color='Survived')

fig2=px.histogram(titanic_df,
             x='Pclass',
             barmode="group",
             marginal="box", #nbins=50,
             color='Survived')

fig3=px.histogram(titanic_df,
             x='Embarked',
             barmode="group",
             marginal="box", #nbins=50,
             color='Survived')

fig4=px.histogram(titanic_df,
             x='SibSp',
             barmode="group",
             marginal="box", #nbins=50,
             color='Survived')

fig5=px.histogram(titanic_df,
             x='Parch',
             barmode="group",
             marginal="box", #nbins=50,
             color='Survived')

fig = make_subplots(rows=1, cols=5, subplot_titles=('Sex', 'Pclass', 'Embarked', 'SibSp', 'Parch'))

fig.add_trace(fig1.data[0], row=1, col=1)
fig.add_trace(fig1.data[2], row=1, col=1)
fig.add_trace(fig2.data[0], row=1, col=2)
fig.add_trace(fig2.data[2], row=1, col=2)
fig.add_trace(fig3.data[0], row=1, col=3)
fig.add_trace(fig3.data[2], row=1, col=3)
fig.add_trace(fig4.data[0], row=1, col=4)
fig.add_trace(fig4.data[2], row=1, col=4)
fig.update_traces(showlegend=False)
fig.add_trace(fig5.data[0], row=1, col=5)
fig.add_trace(fig5.data[2], row=1, col=5)
fig.update_yaxes(title_text="Amount", row=1, col=1)

fig.update_layout(width=900, height=300, title_text='OBSERVATION #1 WITH RESPECT TO SURV/NOT SURV STATUS') 
fig.update_layout(margin=dict(l=0, r=0, t=100, b=0))
fig.show()

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio # без этого фигуры ниже не отображаются
pio.renderers.default = "iframe"

fig1=px.histogram(titanic_df,
             x='Age',
             #color_discrete_sequence=px.colors.sequential.RdBu, 
             barmode="group",
             marginal="box", nbins=100,
             color='Survived'
                )


fig2=px.histogram(titanic_df,
             x='Fare',
             #color_discrete_sequence=px.colors.sequential.RdBu, 
             barmode="group",
             marginal="box", #nbins=50,
             color='Survived')


fig = make_subplots(rows=2, cols=2, column_widths=[0.4, 0.6], row_heights=[0.3, 0.7], subplot_titles=('Age', 'Fare'))

fig.add_trace(fig1.data[1], row=1, col=1)
fig.add_trace(fig1.data[-1], row=1, col=1)
fig.add_trace(fig1.data[0], row=2, col=1)
fig.add_trace(fig1.data[2], row=2, col=1)
fig.update_traces(showlegend=False)
fig.update_xaxes(title_text="Age", row=2, col=1)
fig.update_yaxes(title_text="Amount", row=2, col=1)

fig.add_trace(fig2.data[1], row=1, col=2)
fig.add_trace(fig2.data[-1], row=1, col=2)
fig.add_trace(fig2.data[0], row=2, col=2)
fig.add_trace(fig2.data[2], row=2, col=2)
fig.update_xaxes(title_text="Fare", row=2, col=2)
fig.update_yaxes(title_text="Amount", row=2, col=2)

fig.update_layout(width=900, height=400, title_text='OBSERVATION #2 WITH RESPECT TO SURV/NOT SURV STATUS') 
fig.update_layout(margin=dict(l=0, r=0, t=100, b=0))
fig.show()

#%matplotlib inline
# посмотрим корреляцию между признаками и выжтванием
plt.figure(figsize=(8,6))
ax = plt.axes()

train_corr=titanic_df.copy().drop('PassengerId', axis=1)

# переведем категории в цифры
train_corr.Sex=lb_make.fit_transform(train_corr.Sex)
train_corr.Embarked=lb_make.fit_transform(train_corr.Embarked) 

features_corr = sns.heatmap(train_corr.corr(), annot=True, cmap='coolwarm') 
features_corr=features_corr.set_yticklabels(features_corr.get_yticklabels(), rotation=0, horizontalalignment='right')
ax.set_title('OBSERVATION #3 FEATURES CORRELATION')
plt.show()
"""**Resume by observation:**
- выживание тесно связано с полом пассажира
- у женщин больше шансов выжить
- чем ниже класс каюты, тем меньше шансов выжить
- посадка в порту С ведет к бо'льшим шансам выжить
- с ростом кол-ва SibSp шансы выжить уменьшаются
- по возрасту м-у погибшими и выжившими значимых разлиий нет
- но маленькие дети выживают лучше всех
- тарифы: медиана тарифов выживших значительно отличается от медианы погибших


**Features recognition:**
- пол, класс каюты, порт посадки, тарифы могут быть использованы  в качестве признаков для обучения модели
- возможно стоит как-то перобразовать SibSp и Parch, чтобы усилить связь со статусом пассажира
- попробовать перевести возраст в категории и посмотреть, как повыситься предсказывание модели"""

#**FILLING IN MISSED DATA.AGE**
for dataset in data:
    dataset['Title'] = dataset.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
# для создания словаря титулов нужно убедиться, что наборы титулов в двух df не отличаются
from deepdiff import DeepDiff

trd_titles = trd.groupby(['Pclass']).Title.apply(set).to_dict()
ted_titles = ted.groupby(['Pclass']).Title.apply(set).to_dict()

diff = DeepDiff(trd_titles, ted_titles)
diff['set_item_added']
# как видим, trd не содержит Pclass 1: Dona, Pclass 3: MS - нужно это учесть при составлении словаря титулов

# vocab for the age-gap replacement
mean_age_df = trd.groupby(['Pclass','Sex','Title'], as_index=False).agg({'PassengerId':'count', 'Age':'mean'}).rename(columns={'PassengerId':'Amount', 'Age':'mean_Age'}).fillna(trd.Age.mean())
std_age = trd.groupby(['Pclass','Sex','Title'], as_index=False).Age.std().rename(columns={'Age':'std_Age'}).fillna(0)
mean_age_df = mean_age_df.merge(std_age, how='outer', on=['Pclass','Sex','Title'])
# добавляем недостающие титулы
dona_1 = trd.loc[(trd.Pclass==1)&(trd.Sex=='female'), 'Age'].mean()
ms_3 = trd.loc[(trd.Pclass==3)&(trd.Sex=='female'), 'Age'].mean()
mean_age_df = mean_age_df.append(pd.DataFrame([[1,'female', 'Dona', 1, dona_1, 0],[3,'female', 'Ms', 1, ms_3, 0]], 
                                              columns=['Pclass','Sex','Title','Amount','mean_Age','std_Age']), ignore_index=True)
mean_age_df.head()

f AgeFillIn(df):
        """ 
        Ф-я возвращает возраст:
        Если возраст в ячейке пропущен ф-я берет данные из диапазона: 
        ср. возраст по классу каюты, полу, титулу  +/- ст. откл,
        в противном случае оставляет данные как есть 
        """
        Age, pclass, sex, title = df[0], df[1], df[2], df[3]

        if pd.isnull(Age):
            filter_data = (mean_age_df.Pclass==pclass)&(mean_age_df.Sex==sex)&(mean_age_df.Title==title)
            res = mean_age_df.loc[filter_data, ['mean_Age', 'std_Age']]
            mean_age, std_age = res.mean_Age, res.std_Age
            age = [uniform.rvs(mean_age+std_age, mean_age-std_age)][0]
            return age
        else:
            return Age
for dataset in data:
    dataset['Age_new'] = dataset[['Age','Pclass','Sex','Title']].apply(AgeFillIn, axis = 1)
# check for missing age data
trd.Age_new.isnull().sum(), ted.Age_new.isnull().sum()
# mean age after the missed data was filled in
trd.Age_new.mean(), ted.Age_new.mean()
# check for missing data (Age column will be dropped)
train_missing = trd.isnull().sum().to_frame().reset_index()
test_missing = ted.isnull().sum().to_frame().reset_index()
missing_data = train_missing.merge(test_missing, on='index').rename(columns={'0_x':'train', '0_y':'test'})
missing_data
#Заполним оставшиеся пропуски позже при помощи SimpleImputer

# **II. FEATURES' DESIGN**
# **FAMILY**
"""- наличие родственников на борту повышает шансы выжит
- разделим пассажиров на 3 категории: одинокие/с семьей до 4 человек/остальное - Family1
- разделим пассажирова на 2 категории: одинокие/с семьей - Family2
"""
for dataset in data:
    dataset['Family'] = dataset['SibSp'] +dataset['Parch']
    dataset['Family2'] = pd.cut(dataset.Family, bins=[-1,0, 4, 10], labels=['Alone','1-4', '>4'])
    dataset['Family1'] = pd.cut(dataset.Family, bins=[-1,0,10], labels=['Alone','withFamily'])
trd.groupby('Family1', as_index=False).agg({'PassengerId':'count', 'Survived':'mean'}).rename(columns={'PassengerId':'Amount'})
trd.groupby('Family2', as_index=False).agg({'PassengerId':'count', 'Survived':'mean'}).rename(columns={'PassengerId':'Amount'})
#**AGE**
"""
- нужно помнить, что отсуствующие данные были компенсированы
- маленькие дати до 5 лет выживают лучше 
- для того, чтобы не зависеть от искусственных данных переведем возраст в следующие 2 вида:
    - разделим пассажиров на 3 категории: дети до 5 лет, те у кого есть данные, остальные - Age_cat1
    - разделим пассажиров на 2 категории: дети до 5 лет и остальные - Age_cat2
 """
 # распределение известных  данных по возрасту с учетом пола и статуса выживания
fig_dims = (14, 2.5)
fig = plt.subplots(figsize=fig_dims) 
sns.scatterplot(x=titanic_df["Age"], y=titanic_df["Sex"], hue=titanic_df["Survived"], style=titanic_df['Sex'])
plt.show()
# основано на данных до компенсации возраста
#titanic_df['Age_category'] = pd.cut(titanic_df.Age, bins=[0,5,90], labels=['Kid','Others']) 
titanic_df['Age_category']  = titanic_df.Age.apply(lambda x: np.where(x <=5, 'Kid', np.where(x>5, 'YES', 'NO')))
titanic_df.groupby(['Age_category'], as_index=False).agg({'PassengerId':'count', 'Survived':'mean'}).rename(columns={'PassengerId':'Amount'})
# Дети до 5 выживают значительно лучше, чем все остальные
# применим 2 варианта
for dataset in data:
    dataset['Age_cat1'] = dataset.Age.apply(lambda x: np.where(x <=5, 'Kid', np.where(x>5, 'Yes', 'No')))
    dataset['Age_cat2'] = pd.cut(dataset.Age_new, bins=[0,5,90], labels=['Kids','Others']) 
#**FARE**
"""
- мы видели выше, что погибло очень много пассажиров с дешевыми билетами
- создадим 3 категории пассажиров: с билетами до 11, до 50 и остальные
"""
fig_dims = (14, 4)
fig, ax = plt.subplots(figsize=fig_dims)
sns.scatterplot(x=titanic_df["Age"], y=titanic_df["Fare"], hue=titanic_df["Survived"], style=titanic_df['Sex'])
ax.set_ylim(-1, 80)
plt.show()
# основано на данных до компенсации возраста, показаны только значения Fare в диапазоне  0-80
#**Покажем, что между Survived и Fare существует взаимосвязь**
fare = trd.copy()[['Fare', 'Survived']]
fare.loc[:,'Survived'].replace([0,1],['No','Yes'], inplace=True)
fare.hist(column="Fare",by="Survived",sharey=True);
fare.loc[:,'fare_groups'] = pd.cut(fare.Fare, bins=[-1,11,50, 515], labels=['Chip','Mid', 'Others'])
fare['Count'] = 1
fare_grouped = fare.groupby(['Survived', 'fare_groups'], as_index=False).Count.sum()
fare_observation = fare_grouped.pivot(index='Survived', columns='fare_groups', values='Count')
fare_observation 
"""
H_0: the observed frequencies of two categories in the table are independent (there is no relationship between the Survived and Fare variables)

H_1: the observed frequencies of two categories in the table are dependent (there is relationship between the Survived and Fare variables)
"""
from scipy.stats import chi2_contingency

# Chi-square statistic
chi2, p_value, degree_of_freedom, expected = chi2_contingency(fare_observation)

print('chi2:{}\ndegree_of_freedom:{}\np-value:{}'.format(chi2,degree_of_freedom,p_value))
# p-value = 1.3*10^(-24) < 0.05 --> we regect H_0 --> there is relationship between the Survived and Fare variables
trd['Fare_category'] = pd.cut(trd.Fare, bins=[-1,11,50, 515], labels=['Chip','Mid', 'Others']) 
trd.groupby(['Fare_category'], as_index=False).agg({'PassengerId':'count', 'Survived':'mean'}).rename(columns={'PassengerId':'Amount'})
# видим, что пассажиры с билетами стоимостью до 11$ имеют шансы выжить хуже
# применим 3 билетные группы:билеты до 11$, 50$  и остальные
for dataset in data:
    dataset['Fare_category'] = pd.cut(dataset.Fare, bins=[-1,11,50, 515], labels=['Chip','Mid', 'Others']) 
#**Перед выбором признаков посмотрим на корреляционную матрицу**
# посмотрим что дали дополнительные признаки
plt.figure(figsize=(10,7))
ax = plt.axes()

train_corr=trd.copy().drop('PassengerId', axis=1)

# переведем категории в цифры
train_corr.Sex=lb_make.fit_transform(train_corr.Sex)
train_corr.Embarked=lb_make.fit_transform(train_corr.Embarked) 
train_corr.Family1=lb_make.fit_transform(train_corr.Family1)
train_corr.Family2=lb_make.fit_transform(train_corr.Family2) 
train_corr.Age_cat1=lb_make.fit_transform(train_corr.Age_cat1)
train_corr.Age_cat2=lb_make.fit_transform(train_corr.Age_cat2) 
train_corr.Fare_category=lb_make.fit_transform(train_corr.Fare_category) 

features_corr = sns.heatmap(train_corr.corr(), annot=True, cmap='coolwarm') 
features_corr=features_corr.set_yticklabels(features_corr.get_yticklabels(), rotation=0, horizontalalignment='right')
ax.set_title('OBSERVATION #4 FEATURES CORRELATION')
plt.show()
"""
Оставим для обучения признаки с наибольшим значениями в матрице корреляции по отношению к Survived:

'Sex','Pclass','Fare','Fare_category','Embarked','Age_new','Family1','Family2','Age_cat2'
"""
trd.info()
for dataset in data:
    dataset.drop(['Name', 'Title', 'Age', 'SibSp', 'Parch', 'Family', 'Age_cat1'], axis=1, inplace=True)
    dataset['Sex'] = dataset.Sex.astype('category')
    dataset['Embarked'] = dataset.Embarked.astype('category')
    dataset['Pclass'] = dataset.Pclass.astype('category')
trd.info()
# check for missing data
train_missing = trd.isnull().sum().to_frame().reset_index()
test_missing = ted.isnull().sum().to_frame().reset_index()
missing_data = train_missing.merge(test_missing, on='index').rename(columns={'0_x':'train', '0_y':'test'})
missing_data
#пропущенные данные будут заполнены дальше