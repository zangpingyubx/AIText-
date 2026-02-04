import pandas as pd

# 创建一个简单的Series（指定数据和索引）
s = pd.Series(
    data=[100, 80, 90, 75],  # 数据
    index=["小明", "小红", "小刚", "小丽"],  # 自定义索引（行标签）
    name="数学成绩" # 系列名称（对应列名）

)

# 查看Series
print(s)
print("-" * 30)
# 通过索引获取数据
print("小明的数学成绩：", s["小明"])



# 用字典创建DataFrame（字典的键为列名，值为列数据）
df = pd.DataFrame(
    data={
        "数学": [100, 80, 90, 75],
        "语文": [95, 85, 88, 92],
        "英语": [90, 95, 85, 80]
    },
    index=["小明", "小红", "小刚", "小丽"]  # 自定义行索引
)

# 查看DataFrame
print("完整表格数据：")
print(df)
print("-" * 30)
# 查看列索引
print("列索引：", df.columns)
print("-" * 30)
# 提取「语文」列的数据（返回一个Series）
print("语文成绩列：")
print(df["语文"])
print("-" * 30)
# 提取小红的所有成绩
print("小红的各科成绩：")
print(df.loc["小红"])


#读取 csv 结构化文件
df=pd.read_csv("E:\\AI学习\\测试\\45004_2021010100.csv")
print(df)
df.to_csv("E:\\AI学习\\测试\\45004_2021010100sc.csv")
print("-" * 30)
# 读取excel文件
excel_df=pd.read_excel("E:\\AI学习\\测试\\统计快报表.xlsx")
print(excel_df)
# 输出excel文件
excel_df.to_excel("E:\\AI学习\\测试\\统计快报表.xlsx")
print("-" * 30)
# 读取JSON、SQL、HTML、Parquet 文件
json_df=pd.read_json("E:\\AI学习\\测试\\endpoint.json")
print("json内容")
print(json_df)
#输出json、SQL、HTML、Parquet 文件等
json_df.to_json("E:\\AI学习\\测试\\endpointfz.json")
print("-" * 100)




"""
手动解析json解析本地 AWS 端点配置 JSON 文件
:param file_path: JSON 文件路径
:return: 解析后的 Python 数据结构（字典）
"""
import json
try:
    # 1. 打开并读取 JSON 文件（指定 utf-8 编码避免乱码）
    with open("E:\\AI学习\\测试\\endpointTo.json", "r", encoding="utf-8") as f:
        # 2. 将 JSON 内容转为 Python 字典（核心解析步骤）
        json_data = json.load(f)

        # 3. 提取核心信息（和场景 1 一致，此处仅展示关键提取）
        print("解析成功！顶层版本：", json_data["version"])
        print("必填参数 UseDualStack 的默认值：", json_data["parameters"]["UseDualStack"]["default"])
        # 4. 修正：逐层遍历列表+字典，提取所有 ref 对应的值（核心修改）
        print("所有 ref 对应的取值：")
        # 第一步：遍历 rules 列表（列表，用整数索引/for 循环遍历）
        for rule in json_data["rules"]:
            # 第二步：遍历当前规则中的 conditions 列表（列表，继续 for 循环）
            for condition in rule.get("conditions", []):  # 用 get 避免键不存在报错
                # 第三步：遍历当前条件中的 argv 列表（列表，继续 for 循环）
                for argv_item in condition.get("argv", []):
                    # 第四步：argv_item 是字典，用 ["ref"] 访问取值（字典可用字符串键名）
                    if "ref" in argv_item:  # 先判断是否有 ref 键，避免 KeyError
                        ref_value = argv_item["ref"]
                        print(f"  ref: {ref_value}")

        # 5. （可选）打印完整解析结果（数据量较大，可注释掉）
        print("完整解析结果：", json_data)
except FileNotFoundError:
    print(f"错误：未找到文件")
except json.JSONDecodeError:
    print("错误：JSON 格式无效，无法解析")
print("-" * 100)


"""
pandas数据清洗

便捷的数据读取与写入
支持直接读写多种常见数据格式，无需手动解析，这是数据处理的第一步：
结构化文件：CSV（pd.read_csv()/df.to_csv()）、Excel（pd.read_excel()/df.to_excel()，需安装openpyxl依赖）。
其他格式：JSON、SQL、HTML、Parquet 等。
示例：df = pd.read_csv("学生成绩表.csv") 一键读取 CSV 文件为 DataFrame。

强大的数据清洗能力
真实数据往往存在缺失、重复、异常等问题，Pandas 提供了一站式清洗工具：
缺失值处理：df.dropna()（删除缺失值）、df.fillna()（填充缺失值，如用均值、中位数填充）。
重复值处理：df.duplicated()（检测重复值）、df.drop_duplicates()（删除重复值）。
数据类型转换：df.astype()（转换列数据类型）、pd.to_datetime()（转换为日期类型）。
异常值筛选：通过条件索引快速过滤异常数据（如df[df["数学"] > 60]筛选数学及格的学生）。

灵活的数据筛选与索引
支持多种方式快速定位和提取所需数据，比 Excel 的筛选功能更高效：
列提取：df["列名"] 或 df.loc[:, "列名"]（提取指定列）。
行提取：df.loc["行索引"]（按标签提取）、df.iloc[行号]（按位置提取）。
条件筛选：df[df["数学"] >= 90]（筛选数学 90 分及以上的学生），支持多条件组合。

高效的数据转换与聚合
支持对数据进行分组、合并、透视等操作，用于挖掘数据的内在规律：
新增列：df["总分"] = df["数学"] + df["语文"] + df["英语"]（一键计算总分列）。
分组聚合：df.groupby("班级")["数学"].mean()（按班级分组，计算各班数学平均分），对应 Excel 的「数据透视表」。
数据合并：pd.merge()（类似 SQL 的 JOIN 操作，合并两个表格）、pd.concat()（拼接多个表格）。

便捷的数据统计与可视化
内置统计函数：df.describe()（一键生成数据的均值、方差、最值、分位数等统计信息）、df.corr()（计算列之间的相关系数）。
无缝对接可视化库：直接调用df.plot()（基于 Matplotlib），快速绘制折线图、柱状图、直方图等，无需手动转换数据格式。
"""
#读取 csv 结构化文件
df=pd.read_csv("E:\\AI学习\\测试\\45004_2021010100.csv")
#删除缺失值
# (how="all"所有列或行为None时才删除，any只要某一行或某一列中有None值就删除,
# axis=1（检查范围0行，1列）,
# inplace=True 修改DataFrame,False新建DataFrame
df.dropna(how="all", axis=1, inplace=True)
df.dropna(how="all", axis=0, inplace=True)
print("删除全是None的列，删除全是None的行，操作同一个df,没有新建\n",df)
