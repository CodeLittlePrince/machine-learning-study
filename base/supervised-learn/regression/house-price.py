# 一元线性回归
import numpy as np
from matplotlib import pyplot as plt

x = np.array([56, 72, 69, 88, 102, 86, 76, 79, 94, 74])
y = np.array([92, 102, 86, 110, 130, 99, 96, 102, 105, 92])

plt.scatter(x, y)
plt.xlabel('Area')
plt.ylabel('Price')
# plt.show()

# 线性方程
def f(x, w0, w1):
    y = w0 + w1 * x
    return y

# 平方损失函数
def square_loss(x, y, w0, w1):
    loss = sum(np.square(y - (w0 + w1*x)))
    return loss

# 最佳拟合直线
def w_calculator(x, y):
    n = len(x)
    w1 = (n*sum(x*y) - sum(x)*sum(y))/(n*sum(x*x) - sum(x)*sum(x))
    w0 = (sum(x*x)*sum(y) - sum(x)*sum(x*y))/(n*sum(x*x)-sum(x)*sum(x))
    return w0, w1

w0, w1 = w_calculator(x, y)

# loss = square_loss(x, y, w0, w1)

# x_temp = np.linspace(50, 120, 100)  # 绘制直线生成的临时点
# plt.plot(x_temp, x_temp * w1 + w0, 'r')
# plt.show()
print('===== 代数推导实现')
print(w0, w1)
print(f(150, w0, w1))


# ====== 以上都为代数偏导数求解的推导实现，下面使用矩阵来实现
def w_matrix(x, y):
    w = (x.T * x).I * x.T * y
    return w

x = np.matrix([[1, 56], [1, 72], [1, 69], [1, 88], [1, 102],
               [1, 86], [1, 76], [1, 79], [1, 94], [1, 74]])
y = np.matrix([92, 102, 86, 110, 130, 99, 96, 102, 105, 92])
w0, w1 = w_matrix(x, y.reshape(10, 1))

print('===== 矩阵推导实现')
print(w0, w1)
print(f(150, w0, w1))

# ====== 以上都为纯数学的推导实现，下面使用sklearn来实现
from sklearn.linear_model import LinearRegression

# 定义线性回归模型
x = np.array([56, 72, 69, 88, 102, 86, 76, 79, 94, 74])
y = np.array([92, 102, 86, 110, 130, 99, 96, 102, 105, 92])

model = LinearRegression()
model.fit(x.reshape(len(x), 1), y)  # 训练, reshape 操作把数据处理成 fit 能接受的形状

# 得到模型拟合参数
print('===== sklearn实现')
print(model.intercept_, model.coef_)
print(model.predict([[150]]))