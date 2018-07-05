# coding: utf-8
import pdb
import pandas as pd  # pandas是python的数据格式处理类库
from abupy import AbuML
import sklearn.preprocessing as preprocessing

"""DATA"""


"""function"""


def set_missing_ages(p_df):
    """处理数据缺失"""
    p_df.loc[(p_df.Age.isnull()), 'Age'] = p_df.Age.dropna().mean()
    return p_df


def fit_transform(data_train, df):
    """归一化数据"""
    scaler = preprocessing.StandardScaler()
    df['Age_scaled'] = scaler.fit_transform(data_train['Age'].values.reshape(-1, 1))
    df['Fare_scaled'] = scaler.fit_transform(data_train['Fare'].values.reshape(-1, 1))
    return data_train


def set_cabin_type(p_df):
    """处理特别类型数据"""
    p_df.loc[(p_df.Cabin.notnull()), 'Cabin'] = "Yes"
    p_df.loc[(p_df.Cabin.isnull()), 'Cabin'] = "No"
    return p_df


def reshape(data_train, df):
    """重组数据"""
    # 处理特别类型数据
    dummies_pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')  # 分级数据
    dummies_embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
    dummies_sex = pd.get_dummies(data_train['Sex'], prefix='Sex')

    # 重组数据维度
    df = pd.concat([df, dummies_embarked, dummies_sex, dummies_pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    return df


def new(df):
    # 选择哪些特征作为训练特征
    train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_df.head(1)

    # 用新加入模型训练
    train_np = train_df.as_matrix()
    y = train_np[:, 0]
    x = train_np[:, 1:]
    new_titanic = AbuML(x, y, train_df)

    return new_titanic

"""common"""


"""work"""


class Titanic(object):
    def __init__(self):
        self.titanic = AbuML.create_test_more_fiter()
        self.data_train = pd.read_csv("./data/titanic/train.csv")
        self.df = None
        # data_train.info()

    def pre(self):
        # 处理数据缺失
        self.df = set_missing_ages(self.data_train)
        # 归一化
        self.data_train = fit_transform(self.data_train, self.df)
        # 处理个别数据
        self.df = reshape(self.data_train, self.df)

    def show(self):
        self.data_train.head(10)
        self.titanic.plot_confusion_matrices()  # 混淆矩阵
        self.titanic.plot_roc_estimator()  # roc曲线
        self.data_train.groupby('Survived').count()  # 分组计数
        self.titanic.estimator.logistic_classifier()  # 有监督学习分类器
        self.titanic.cross_val_accuracy_score()  # 有监督学习分类
        self.titanic.importances_coef_pd()  # 评估特征作用


"""show"""


if __name__ == '__main__':
    tan = Titanic()
    tan.pre()
    pdb.set_trace()
    print(tan)