# 多项式回归
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import leastsq
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

x = [4, 8, 12, 25, 32, 43, 58, 63, 69, 79]
y = [20, 33, 50, 56, 42, 31, 33, 46, 65, 75]

# plt.scatter(x, y)
# plt.show()

def func(p, x):
    # 根据公式，定义 2 次多项式函数
    w0, w1, w2 = p
    f = w0 + w1*x + w2*x*x
    return f


def err_func(p, x, y):
    # 残差函数（观测值与拟合值之间的差距）
    ret = func(p, x) - y
    return ret

p_init = np.random.randn(3)  # 生成 3 个随机数
# 使用 Scipy 提供的最小二乘法函数得到最佳拟合参数
parameters = leastsq(err_func, p_init, args=(np.array(x), np.array(y)))

print('Fitting Parameters: ', parameters[0])

# 绘制拟合图像时需要的临时点
x_temp = np.linspace(0, 80, 10000)

# 绘制拟合函数曲线
# plt.plot(x_temp, func(parameters[0], x_temp), 'r')

# 绘制原数据点
# plt.scatter(x, y)

def fit_func(p, x):
    # 根据公式，定义 n 次多项式函数
    f = np.poly1d(p)
    return f(x)


def err_func(p, x, y):
    # 残差函数（观测值与拟合值之间的差距）
    ret = fit_func(p, x) - y
    return ret


def n_poly(n):
    # n 次多项式拟合
    p_init = np.random.randn(n)  # 生成 n 个随机数
    parameters = leastsq(err_func, p_init, args=(np.array(x), np.array(y)))
    return parameters[0]

# n_poly(3)

# 绘制拟合图像时需要的临时点
# x_temp = np.linspace(0, 80, 10000)

# 绘制子图
# fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# axes[0, 0].plot(x_temp, fit_func(n_poly(4), x_temp), 'r')
# axes[0, 0].scatter(x, y)
# axes[0, 0].set_title("m = 3")

# axes[0, 1].plot(x_temp, fit_func(n_poly(5), x_temp), 'r')
# axes[0, 1].scatter(x, y)
# axes[0, 1].set_title("m = 4")

# axes[0, 2].plot(x_temp, fit_func(n_poly(6), x_temp), 'r')
# axes[0, 2].scatter(x, y)
# axes[0, 2].set_title("m = 5")

# axes[1, 0].plot(x_temp, fit_func(n_poly(7), x_temp), 'r')
# axes[1, 0].scatter(x, y)
# axes[1, 0].set_title("m = 6")

# axes[1, 1].plot(x_temp, fit_func(n_poly(8), x_temp), 'r')
# axes[1, 1].scatter(x, y)
# axes[1, 1].set_title("m = 7")

# axes[1, 2].plot(x_temp, fit_func(n_poly(9), x_temp), 'r')
# axes[1, 2].scatter(x, y)
# axes[1, 2].set_title("m = 8")

# plt.show()

# ====== 以上都为代数求解的推导实现，下面使用sklearn来实现
x = [4, 8, 12, 25, 32, 43, 58, 63, 69, 79]
y = [20, 33, 50, 56, 42, 31, 33, 46, 65, 75]
x_reshape = np.array(x).reshape(len(x), 1)  # 转换为列向量
y_reshape = np.array(y).reshape(len(y), 1)  # 转换为列向量

# 使用 sklearn 得到 3 次多项式回归特征矩阵
poly_features = PolynomialFeatures(degree=3, include_bias=False)
poly_x = poly_features.fit_transform(x_reshape)
poly_y = poly_features.fit_transform(y_reshape)
print(poly_x)

# 定义线性回归模型
model = LinearRegression()
model.fit(poly_x, y)  # 训练

# 绘制拟合图像
x_temp = np.linspace(0, 80, 10000)
x_temp = np.array(x_temp).reshape(len(x_temp), 1)
poly_x_temp = poly_features.fit_transform(x_temp)

plt.plot(x_temp, model.predict(poly_x_temp), 'r')
plt.scatter(x, y)
plt.show() # 发现和之前代数求解m=4的结果一致