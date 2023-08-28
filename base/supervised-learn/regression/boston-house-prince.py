# 多元线性回归
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('./boston-house-prince.csv')
# print(df.head())

features = df[['crim', 'rm', 'lstat']] # 特征值数据选取

target = df['medv']  # 目标值数据

split_num = int(len(features)*0.7)  # 得到 70% 位置

x_train = features[:split_num]  # 训练集特征
y_train = target[:split_num]  # 训练集目标

X_test = features[split_num:]  # 测试集特征
y_test = target[split_num:]  # 测试集目标

model = LinearRegression()  # 建立模型
model.fit(x_train, y_train)  # 训练模型 fit译为拟合
print('输出训练后的模型参数和截距项: ')
print(model.coef_, model.intercept_)  # 输出训练后的模型参数和截距项 [0.69979497 10.13564218 -0.20532653] -38.000969889690296
preds = model.predict(X_test)  # 预测结果
print('输出预测结果: ')
print(preds)

# ====== 结果评估

# 平均绝对误差（MAE）
def mae_value(y_true, y_pred):
    n = len(y_true)
    mae = sum(np.abs(y_true - y_pred))/n
    return mae

# 均方误差（MSE）
def mse_value(y_true, y_pred):
    n = len(y_true)
    mse = sum(np.square(y_true - y_pred))/n
    return mse

mae = mae_value(y_test.values, preds)
mse = mse_value(y_test.values, preds)

print("MAE: ", mae)
print("MSE: ", mse)
# 误差那么大的原因：
# 1、没有对数据预处理；
# 2、利用的特征太少，只有 3 个；