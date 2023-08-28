import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

df = pd.read_csv('./vaccine.csv')
# 定义 x, y 的取值
x = df['Year']
y = df['Values']
# 绘图
plt.plot(x, y, 'r')
# plt.scatter(x, y)
# plt.show()

# 首先划分 dateframe 为训练集和测试集
train_df = df[:int(len(df)*0.7)]
test_df = df[int(len(df)*0.7):]

# 定义训练和测试使用的自变量和因变量
X_train = train_df['Year'].values
y_train = train_df['Values'].values

X_test = test_df['Year'].values
y_test = test_df['Values'].values

# 建立线性回归模型
model = LinearRegression()
model.fit(X_train.reshape(len(X_train), 1), y_train.reshape(len(y_train), 1))
results = model.predict(X_test.reshape(len(X_test), 1))
print(results)  # 线性回归模型在测试集上的预测结果
print("线性回归平均绝对误差: ", mean_absolute_error(y_test, results.flatten()))
print("线性回归均方误差: ", mean_squared_error(y_test, results.flatten()))

def n_poly(n):
    # n 次多项式回归特征矩阵
    poly_features = PolynomialFeatures(degree=n, include_bias=False)
    poly_X_train = poly_features.fit_transform(
        X_train.reshape(len(X_train), 1))
    poly_X_test = poly_features.fit_transform(X_test.reshape(len(X_test), 1))

    # n 次多项式回归模型训练与预测
    model = LinearRegression()
    model.fit(poly_X_train, y_train.reshape(len(X_train), 1))  # 训练模型

    results = model.predict(poly_X_test)  # 预测结果
    print(results)
    print(n, "次多项式回归平均绝对误差: ", mean_absolute_error(y_test, results.flatten()))
    print(n, "次多项式回归均方误差: ", mean_squared_error(y_test, results.flatten()))

# 1 次多项式回归模型训练与预测 (理论上和线性回归是一模一样的，实际确实如此)
n_poly(1)

# 2 次多项式回归模型训练与预测
n_poly(2)

# 3 次多项式回归模型训练与预测
n_poly(3)
