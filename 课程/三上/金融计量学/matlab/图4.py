import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.tseries.offsets import QuarterBegin
import matplotlib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 读取 Excel 文件
file_path = 'E:\\jupyter_program\\课程\\金融计量学\\matlab\\data_origin.xlsx'
data = pd.read_excel(file_path)

# 生成日期列
start_date = pd.to_datetime('1997-01-01')
data['日期'] = [start_date + i * QuarterBegin(1) for i in range(len(data))]

data['Export增长率'] = data['Export']/100
# 定义要绘制的列
columns_to_plot = ['Export增长率', 'Rate007']

# 定义颜色
colors = ['blue', 'orange']

# 创建图形和轴
plt.figure(figsize=(12, 6))

# 绘制每条折线
for i, column in enumerate(columns_to_plot):
    plt.plot(data['日期'], data[column], '-', color=colors[i], label=column)

# 标题和标签
plt.ylabel('%')

# 设置横轴日期格式和刻度
years = mdates.YearLocator()
years_fmt = mdates.DateFormatter('%Y')
plt.gca().xaxis.set_major_locator(years)
plt.gca().xaxis.set_major_formatter(years_fmt)

quarters = mdates.MonthLocator(bymonth=[1, 4, 7, 10])
plt.gca().xaxis.set_minor_locator(quarters)

# 旋转主刻度标签
plt.xticks(rotation=45)

# 图例
plt.legend(prop={'size': 20})  # 调整图例字体大小为 12

# 显示图形
plt.tight_layout()
plt.show()