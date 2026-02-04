import json
import pandas as pd
import gzip
import matplotlib.pyplot as plt


def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')



if __name__ == "__main__":
    df=getDF('E:\\AI学习\\礼品卡数据\\Gift_Cards.json.gz')
    print(df)
    # 去一行数据
    print(df.iloc[0])
    #取一列数据
    print(df["overall"])
    #所有列
    print(df.columns)

    """
    清洗数据  删除缺失值
    清洗前：147194 rows x 12 columns
    清洗所有列中有None的数据，出去image列
    清晰后：1416 rows x 12 columns
    """
    exclude_col="image"
    # 步骤 2：提取需要检查缺失值的列列表（所有列 - 排除列）
    check_cols=[col for col in df.columns if col!=exclude_col]
    # (how="all"所有列或行为None时才删除，any只要某一行或某一列中有None值就删除,
    # axis=1（检查范围0行，1列）,
    # inplace=True 修改DataFrame,False新建DataFrame
    # 步骤 3：执行 dropna()，只检查 check_cols 中的列，排除 exclude_col
    df.dropna(axis=0, inplace=True,subset=check_cols)
    print("删除全是None的列，操作同一个df,没有新建\n", df)

    """
    清理reviewText是None的列
    """
    # # 步骤 1：确定要检查缺失值的列（只删除该列有缺失的行）
    # target_col="reviewText"
    # # 步骤 2：执行 dropna()，subset 传入该列（必须是列表格式，即使只有一列）
    # df.dropna(subset=[target_col])
    # print("\n=== 处理后的最终 df（仅删除 reviewText 列有缺失的行）===")
    # print(df)


    """
    统计：
    统计分析（按商品类别统计平均评分、用户评论字数分布）；
    """
    # 步骤 2：按 asin 分组，统计 overall 的平均值
    category_asin_overall = df.groupby("asin")["overall"].mean()
    print("\n=== 按商品类别统计的平均评分（基础结果）===")
    print(category_asin_overall)
    # 优化 1：保留 2 位小数（更符合评分展示习惯）
    category_avg_rating_rounded = category_asin_overall.round(2)

    # 优化 2：转成 DataFrame 类型（列名更清晰，方便后续保存/可视化）
    category_avg_rating_df = category_avg_rating_rounded.reset_index()
    category_avg_rating_df.columns = ["商品类别", "平均评分"]  # 重命名列（可选）

    # 优化 3：按平均评分从高到低排序（更直观）
    category_avg_rating_sorted = category_avg_rating_df.sort_values(by="平均评分", ascending=False)

    print("\n=== 优化后的平均评分结果 ===")
    print(category_avg_rating_sorted)


    """
    可视化（类别评分对比图）；
    """
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # Windows 用 SimHei，Mac 用 Arial Unicode MS
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常
    plt.figure(figsize=(10, 6))  # 设置图表大小（宽 10，高 6）
    # ===================== 步骤 3：绘制柱状图（类别评分对比）=====================
    # 提取 x 轴（类别）、y 轴（平均评分）数据
    x = category_avg_rating_sorted["商品类别"]
    y = category_avg_rating_sorted["平均评分"]
    # 绘制纵向柱状图（颜色可选，这里用渐变色更美观）
    bars = plt.bar(x, y, color=["#1f77b4", "#ff7f0e", "#2ca02c"], alpha=0.8, width=0.6)
    # ===================== 步骤 4：图表美化（添加标签、标题、数值标注，更直观）=====================
    # 4.1 添加坐标轴标签和标题
    plt.title("商品类别平均评分对比图", fontsize=14, pad=20)
    plt.xlabel("商品类别", fontsize=12)
    plt.ylabel("平均评分（星）", fontsize=12)

    # 4.2 设置 y 轴范围（评分 0-5 星，更贴合实际）
    plt.ylim(0, 5.5)

    # 4.3 给每个柱子添加数值标注（显示具体评分）
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2,  # 数值标注的 x 坐标（柱子中间）
                 height + 0.1,  # 数值标注的 y 坐标（柱子上方，避免重叠）
                 f"{height}",  # 标注内容（平均评分）
                 ha="center",  # 水平居中
                 fontsize=10)

    # 4.4 添加网格线（方便读取数值，更整洁）
    plt.grid(axis="y", alpha=0.3, linestyle="--")

    # ===================== 步骤 5：显示图表=====================
    plt.tight_layout()  # 自动调整布局，避免标签重叠
    plt.show()


    """
    评论字数直方图
    """
    # 2.2 定义函数：计算单条评论的字数（中文/英文通用，去除首尾空格）
    def calculate_text_length(text):
        # 转换为字符串，去除首尾空格，返回字符长度
        return len(str(text).strip())


    # 2.3 新增「评论字数」列，应用函数计算
    df["review_word_count"] = df["reviewText"].apply(calculate_text_length)
    # ===================== 步骤 3：可视化配置=====================
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(12, 6))

    # ===================== 步骤 4：绘制直方图=====================
    # 提取评论字数数据
    word_counts = df["review_word_count"]
    # 绘制直方图：bins 设定分组区间（可调整，默认自动分组），edgecolor 增加柱子边框（更清晰）
    n, bins, patches = plt.hist(word_counts, bins=10, edgecolor="black", alpha=0.7, color="#1f77b4")

    # ===================== 步骤 5：图表美化=====================
    # 5.1 添加标签和标题
    plt.title("评论文字数量分布直方图", fontsize=14, pad=20)
    plt.xlabel("评论字数（字符数）", fontsize=12)
    plt.ylabel("评论数量（频数）", fontsize=12)

    # 5.2 添加统计信息（平均字数、中位数，更有参考价值）
    avg_word_count = word_counts.mean().round(2)
    median_word_count = word_counts.median()
    plt.text(0.95, 0.95,  # 文本位置（右上角，相对坐标）
             f"平均字数：{avg_word_count}\n中位数字数：{median_word_count}",
             ha="right", va="top",
             transform=plt.gca().transAxes,  # 相对坐标（不受坐标轴范围影响）
             bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),  # 添加背景框（更醒目）
             fontsize=10)

    # 5.3 添加网格线
    plt.grid(axis="y", alpha=0.3, linestyle="--")

    # ===================== 步骤 6：显示图表=====================
    plt.tight_layout()
    plt.show()






