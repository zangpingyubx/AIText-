"""
Figure：画布，整个绘图窗口的容器，一个画布可以包含 1 个或多个绘图区域。
Axes：绘图区域（子图），真正用于绘制图表的核心区域，所有绘图操作基本都围绕它展开。
常用导入约定：import matplotlib.pyplot as plt（plt是行业通用别名，简化代码书写）。
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # 基于Matplotlib，简化热力图绘制
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

"""
折线图（展示数据趋势，如时间序列、模型训练进度）
"""
# 1. 生成测试数据（0到10之间的100个均匀数，及其正弦值）
x = np.linspace(0, 10, 100)
print(x)
y = np.sin(x)
print(y)

# 2. 创建画布和绘图区域（figsize设置画布宽高，单位为英寸）
fig, ax = plt.subplots(figsize=(8, 4))
#fig 可以设置画布
# fig.subplots_adjust(left=0.13, right=0.85, top=0.9, bottom=0.13)

# 3. 绘制折线图（label用于图例展示，color设置颜色，linewidth设置线宽）
ax.plot(x, y, label="sin(x)", color="blue", linewidth=2)

# 4. 美化图表（必备：标题、坐标轴标签、图例、网格）
ax.set_title("Sine Function Curve", fontsize=14)  # 图表标题
ax.set_xlabel("X Value", fontsize=12)  # X轴标签
ax.set_ylabel("sin(X)", fontsize=12)  # Y轴标签
ax.legend()  # 显示图例（对应plot中的label参数）
ax.grid(True, alpha=0.3)  # 显示网格，alpha设置透明度避免遮挡

# 5. 展示图表（运行后弹出可视化窗口）
plt.show()

# 可选：保存图表（需放在plt.show()之前，否则会保存空白图片）
# fig.savefig("sine_curve.png", dpi=300, bbox_inches="tight")

"""
散点图（展示数据分布 / 变量相关性，如特征与标签的关系）
"""
# 1. 生成随机测试数据（固定随机种子，保证结果可复现）
np.random.seed(42)
x = np.random.randn(500)  # 500个标准正态分布数据
y = np.random.randn(500) + x * 0.5  # y与x存在弱正相关

# 2. 创建画布和绘图区域
fig, ax = plt.subplots(figsize=(8, 6))

# 3. 绘制散点图（s设置点大小，alpha设置透明度避免点重叠）
ax.scatter(x, y, color="red", alpha=0.6, s=20, label="Random Points")

# 4. 美化图表
ax.set_title("Scatter Plot of Correlated Data", fontsize=14)
ax.set_xlabel("X", fontsize=12)
ax.set_ylabel("Y", fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 5. 展示图表
plt.show()


"""
柱状图（展示分类数据对比，如标签分布、模型分类效果）
"""
import matplotlib.pyplot as plt
import numpy as np

# 1. 生成分类测试数据
categories = ["Category A", "Category B", "Category C", "Category D"]
values = [28, 45, 32, 51]

# 2. 创建画布和绘图区域
fig, ax = plt.subplots(figsize=(8, 4))

# 3. 绘制柱状图（width设置柱子宽度，自定义每根柱子颜色）
ax.bar(categories, values, color=["blue", "green", "orange", "red"], width=0.6)

# 4. 美化图表（给柱子添加数值标签，提升可读性）
ax.set_title("Bar Chart of Category Values", fontsize=14)
ax.set_ylabel("Value", fontsize=12)
for i, v in enumerate(values):
    ax.text(i, v + 1, str(v), ha="center", fontsize=10)  # 数值标签居中显示

# 5. 展示图表
plt.show()


"""
数据预处理阶段 - 探索性数据分析（EDA）
"""
# 1. 加载经典鸢尾花分类数据集
iris = load_iris()
X = iris.data  # 4个特征：花萼长/宽、花瓣长/宽
y = iris.target  # 3类鸢尾花标签
feature_names = iris.feature_names
target_names = iris.target_names

# 2. 案例1：绘制特征直方图（查看特征分布，判断是否需要归一化）
fig, ax = plt.subplots(figsize=(10, 4))
for i, target in enumerate(target_names):
    # 提取每类花的「花萼长度」绘制直方图
    ax.hist(X[y == i, 0], alpha=0.6, label=target, bins=15)
ax.set_title("Histogram of Sepal Length (Iris Dataset)", fontsize=14)
ax.set_xlabel(feature_names[0], fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.legend()
plt.show()

# 3. 案例2：特征相关性热力图（分析特征间线性关系，避免多重共线性）
corr = np.corrcoef(X.T)  # 计算特征相关系数矩阵
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", ax=ax, cmap="coolwarm",
            xticklabels=feature_names, yticklabels=feature_names)
ax.set_title("Feature Correlation Heatmap (Iris Dataset)", fontsize=14)
plt.show()

"""
模型训练阶段 - 监控训练过程（Loss / 准确率曲线）
"""
# 1. 模拟训练数据（实际开发中从模型训练日志提取）
epochs = np.arange(1, 51)  # 50个训练轮次
train_loss = np.exp(-epochs/20) + np.random.randn(50)*0.01  # 训练损失（逐渐下降）
val_loss = np.exp(-epochs/25) + np.random.randn(50)*0.02 + 0.05  # 验证损失（后期略升，模拟轻微过拟合）
train_acc = 1 - np.exp(-epochs/15) + np.random.randn(50)*0.01  # 训练准确率（逐渐上升）
val_acc = 1 - np.exp(-epochs/20) + np.random.randn(50)*0.02  # 验证准确率（后期平稳）

# 2. 创建1行2列子图（同时展示Loss和Acc曲线）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 3. 绘制Loss曲线
ax1.plot(epochs, train_loss, label="Train Loss", color="blue", linewidth=2)
ax1.plot(epochs, val_loss, label="Validation Loss", color="red", linewidth=2)
ax1.set_title("Training and Validation Loss", fontsize=14)
ax1.set_xlabel("Epochs", fontsize=12)
ax1.set_ylabel("Loss", fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 4. 绘制Accuracy曲线
ax2.plot(epochs, train_acc, label="Train Accuracy", color="blue", linewidth=2)
ax2.plot(epochs, val_acc, label="Validation Accuracy", color="red", linewidth=2)
ax2.set_title("Training and Validation Accuracy", fontsize=14)
ax2.set_xlabel("Epochs", fontsize=12)
ax2.set_ylabel("Accuracy", fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 5. 展示图表
plt.show()



"""
模型评估阶段 - 分类任务混淆矩阵可视化
"""


# 1. 加载数据并训练简单逻辑回归模型
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 2. 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)
target_names = iris.target_names

# 3. 可视化混淆矩阵
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
ax.figure.colorbar(im, ax=ax)  # 添加颜色条

# 设置坐标轴标签
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=target_names,
       yticklabels=target_names,
       title="Confusion Matrix of Iris Classification Model",
       ylabel="True Label",
       xlabel="Predicted Label")

# 给每个格子添加数值标签
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black")

# 自动调整x轴标签旋转角度
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.show()