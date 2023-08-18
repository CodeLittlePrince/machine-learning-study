# print("Please Input Your Name: ")

# name = input()

# print("Hello " + name)

# from f import greet

# greet('JJ')
# greet('YY', 1)

# def foo(*numbers):
#     sum = 0
#     for n in numbers:
#         sum = sum + n * n
#     return sum

# # print(foo(2,3,4))
# print(foo(*[1,2,3]))

# *args是可变参数，args接收的是一个tuple；
# **kw是关键字参数，kw接收的是一个dict。
# def person(name, age, **kw):
#     if 'city' in kw:
#         # 有city参数
#         pass
#     if 'job' in kw:
#         # 有job参数
#         pass
#     print('name:', name, 'age:', age, 'other:', kw)

# print(person('Jack', 24, city='Beijing', addr='Chaoyang', zipcode=123456))
# def is_odd(n):
#     return n % 2 == 1

# L = list(filter(is_odd, range(1, 20)))
# print(L)

# from PIL import Image 
# img = Image.open('demo.png')
 
# img.show()