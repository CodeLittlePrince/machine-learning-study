import tensorflow as tf

v = tf.Variable([[1, 2], [3, 4]])  # 形状为 (2, 2) 的二维变量

print(v)
print(type(v.numpy()))
print(v.numpy())

print('====')
print(tf.range(start=1, limit=10, delta=2))

print('====')
a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
c = tf.linalg.matmul(a, b)  # 矩阵乘法
print(a)
print(b)
print(c)