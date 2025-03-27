import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']  # 用黑体显示中文

# 构建数据
x = np.arange(5)
# y = [0.893, 0.891, 0.9002, 0.9008, 0.90]
x1 = [78.96, 13.74, 80.32, 80.23, 16.33]
x2 = [25.23, 23.62, 26.85, 26.85, 23.64]


total_width, n = 0.8, 3  # （柱状图的默认宽度值为 0.8）
width = total_width / n

# 绘柱状图
plt.bar(x - width / 2, x1,  width=width, label='Inference time', fc='y')
# plt.bar(x + width / 2, x2, width=width, label='Flops', fc='r')
for a, b in zip(x - width / 2, x1):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

# 在左侧显示图例
plt.legend(frameon=False, loc=(0.73, 0.88))
plt.grid(axis='y', linestyle=':')

# 设置标题
# plt.title("Exploratory experimental results")
# 为两条坐标轴设置名称
plt.xlabel("Various Attention mechanisms")
plt.ylabel("Inference time (ms)")
plt.xticks(np.arange(5), ['SA-solely', 'CA-solely', 'SA+CA (Seq)', 'SA+CA (Par)', 'CCA'])

# 画折线图
ax2 = plt.twinx()
ax2.set_ylabel("Flops (G)")
# ax2.grid(axis='y', linestyle='--')
# 设置坐标轴范围
ax2.set_ylim([20, 30])
# plt.plot(x, y, "r", marker='.', c='black', ms=5, linewidth='1', label="F1-score")
plt.bar(x + width / 2, x2, width=width, label='Flops', fc='r')
# 显示数字
for a, b in zip(x + width / 2, x2):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# 在右侧显示图例
plt.legend(frameon=False, loc=(0.73, 0.83))
plt.savefig("result.jpg", dpi=300)

plt.show()
