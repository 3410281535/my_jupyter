import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']

# 定义文件名和路径
jumpFile = '一年期存款利率.txt'
continuousFile = '七天同业拆借利率.txt'
quarterlyFile = 'M2.txt'

# 定义日期范围
startDate = '1996-08-23'
endDate = '2024-09-30'

# 读取跳跃式日期文件
jumpData = pd.read_csv(jumpFile, delimiter='\t', header=None, names=['日期', '数值'], encoding='utf-8')
jumpData['日期'] = pd.to_datetime(jumpData['日期'], format='%Y-%m-%d')
jumpData = jumpData[(jumpData['日期'] >= startDate) & (jumpData['日期'] <= endDate)]

# 读取连续式日期文件
continuousData = pd.read_csv(continuousFile, delimiter='\t', header=None, names=['日期', '数值'], encoding='utf-8')
continuousData['日期'] = pd.to_datetime(continuousData['日期'], format='%Y-%m-%d')
continuousData = continuousData[(continuousData['日期'] >= startDate) & (continuousData['日期'] <= endDate)]

# 筛选七天同业拆借利率数据中只在月初的点
continuousData_monthly = continuousData[continuousData['日期'].dt.day == 1]

# 读取季度日期文件
quarterlyData = pd.read_csv(quarterlyFile, delimiter='\t', header=None, names=['日期', '数值'], encoding='utf-8')
quarterlyData['日期'] = pd.to_datetime(quarterlyData['日期'] + '-01', format='%Y-%m-%d')
quarterlyData = quarterlyData[(quarterlyData['日期'] >= startDate) & (quarterlyData['日期'] <= endDate)]

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(jumpData['日期'], jumpData['数值'], '-', label='一年期存款利率')  # 只显示线条
plt.plot(continuousData_monthly['日期'], continuousData_monthly['数值'], '-', marker='', label='七天同业拆借利率', zorder=5)  # 只显示月初的折线图，不显示点
plt.plot(quarterlyData['日期'], quarterlyData['数值'], '-', label='M2同比增长率(%)')  # 只显示线条

# 标题和标签

plt.ylabel('%')

# 设置横轴日期格式和刻度
# 使用 YearLocator 每一年设置一个刻度
years = mdates.YearLocator()
years_fmt = mdates.DateFormatter('%Y')
plt.gca().xaxis.set_major_locator(years)
plt.gca().xaxis.set_major_formatter(years_fmt)

# 使用 MonthLocator 每季度设置一个次要刻度（短线小格）
quarters = mdates.MonthLocator(bymonth=[1, 4, 7, 10])
plt.gca().xaxis.set_minor_locator(quarters)
plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter(''))  # 不显示次要刻度的标签

# 不显示次要刻度线（季度分隔线）
plt.grid(False)

# 旋转主刻度标签
plt.xticks(rotation=45)


# 图例
plt.legend(prop={'size': 20})  # 调整图例字体大小为 12

# 显示图形
plt.tight_layout()
plt.show()