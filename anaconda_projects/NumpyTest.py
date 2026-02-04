
import numpy as np
import pandas as pd

# TODO :待完成的任务
# FIXME: 表示此处存在明确的错误、漏洞或不符合预期的逻辑，必须优先修复。
# XXX: 需要优化
# NOTE: 标记重要说明 / 注意事项
# BUG: 标记bug处
# HACK: 不优雅但暂时可用的代码,后续需要重构优化


# 创建一个0-14的数字，将这些数字分成3个组，每个组5个数组
a_numpy=np.arange(15).reshape(3,5)
print('数组:',a_numpy)

# 数组的轴数
print('数组的轴数:',a_numpy.ndim)
# 阵列的尺寸
print('阵列的尺寸:',a_numpy.shape)
# 数组的总元素数
print('数组的总元素数:',a_numpy.size)
# 数组中元素类型的对象
print('数组中元素类型的对象:',a_numpy.dtype)
# 数组中每个元素的字节大小
print('元素的字节大小:',a_numpy.itemsize)
# 包含数组实际元素的缓冲区
print('数组实际元素的缓冲区:',a_numpy.data)
# 多维数组的类型
print('多维数组的类型:',type(a_numpy))


b_numpy=np.arange(15)
print(b_numpy)
print(type(b_numpy))




'''
数组的创建
'''
a = np.array([2, 3, 4])
print('a多维数组的类型:',a.dtype)
b = np.array([1.2, 3.5, 5.1])
print('b多维数组的类型:',b.dtype)

#array将序列序列转换为二维数组， 将序列的序列组成三维数组
b= np.array([(1.5, 2, 3), (4, 5, 6)])
print(b)

#数组的类型也可以在创建时显式指定：
c = np.array([[1, 2], [3, 4]], dtype=complex)
print(c)

"""该函数创建一个充满零的数组，该函数创建一个充满1的数组，而该函数创建一个初始内容随机且依赖于 记忆状态。默认情况下，创建数组的d类型为，
但可以通过关键字参数来指定。zeros ones empty float64 dtype"""
array_numpy=np.zeros((3,4))
print(array_numpy)
#分为两个大组，每个大数组里有3个小数组，每个小数组里有4个全是1的元素
array_numpy1=np.ones((2,3,4),dtype=np.int16)
print(array_numpy1)

# TODO 没搞懂
print("随机",np.empty((2,3)))

#
shuzu_np=np.arange(10,30,5)
print(shuzu_np)

#从0位置开始取，取三位
print(shuzu_np[:3])

#二维及更高维数组可以从嵌套的 Python 初始化序列
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])


"""
数组的排序
"""
#数组排序
arr = np.array([2, 1, 5, 3, 7, 4, 6, 8])
print(np.sort(arr))

#链式
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
print(np.concatenate((a, b)))
x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6]])
print(np.concatenate((x, y),axis=0))

"""
ndarray.ndim会告诉你数组的轴数或维度。

ndarray.size会告诉你数组的总元素数。就是这样 是数组形状中元素的乘积。

ndarray.shape将显示一组整数，表示 元素存储在数组的每个维度上。例如，如果你有 一个有2行3列的二维数组，你的数组形状是。(2, 3)
"""
array_example = np.array([[[0, 1, 2, 3],
                           [4, 5, 6, 7]],

                          [[0, 1, 2, 3],
                           [4, 5, 6, 7]],

                          [[0 ,1 ,2, 3],
                           [4, 5, 6, 7]]])

print("数组的轴数或维度",array_example.ndim)
print("数组中元素的总数",array_example.size)
# 数组的维数 3 每个维度里包含2个小数组，每个小数组里有4个元素 (3, 2, 4)
print("数组形状",array_example.shape)

"""
如何将一维数组转换为二维数组（如何向数组添加新轴）
你可以使用和来增加 的维度 你现有的阵列。np.newaxisnp.expand_dims

使用会使你的数组尺寸增加一个维度 只用一次。这意味着一维数组会变成二维数组，二维数组会变成三维数组，依此类推。np.newaxis
"""


# 创建一个形状为(2, 3, 4)的3维数组（2个3行4列的2维数组）
# 用随机数填充，方便查看差异
arr_3d = np.random.randint(0, 10, size=(2, 3, 4))

# 查看核心信息
print("3维数组的形状：", arr_3d.shape)
print("3维数组的完整内容：\n", arr_3d)
print("-" * 50)

# 创建一个形状为(2, 2, 3, 4)的4维数组
# 2个「2个3行4列」的3维数组
arr_4d = np.random.randint(0, 10, size=(2, 2, 3, 4))

print("-" * 50)
print("4维数组的形状：", arr_4d.shape)
print("提取第1个3维数组的形状：", arr_4d[0].shape)
print("4维数组的完整内容：\n：", arr_4d)

# 创建一个形状为(2, 3, 4, 5,6)的5维数组
# 2个四维数组,每个四维数组里有3个三维维数组,每个三位数组里4个二维数组，每个二维数组里有5个一维数组，每个一维数组里有6个元素
arr_5d = np.random.randint(0, 10, size=(2,3,4,5,6))

print("-" * 50)
print("5维数组的形状：", arr_5d.shape)
print("提取第1个3维数组的形状：", arr_5d[0].shape)
print("5维数组的完整内容：\n：", arr_5d)



# 提取第1个2维数组（索引从0开始，对应「第1叠表格」）
print("第1个2维数组（arr_3d[0]）：\n", pd.DataFrame(list(arr_3d[0]),columns=['a','b','c','d']))
print("-" * 50)

# 提取第2个2维数组中，第1行第2列的数值
print("第2个2维数组的第1行第2列数值：", arr_3d[1, 1, 2])

# 运算
a = np.array([[1,2,3],[4,5,6]])  # shape (2,3)
b = np.array([10,20,30])         # shape (3,)
# b = np.array([10,20,30,40])         # shape (4,) #无法计算
c = a+b  # 广播机制生效，b自动扩展为(2,3)
print("广播值:\n",pd.DataFrame(list(c),columns=['a','b','c']))