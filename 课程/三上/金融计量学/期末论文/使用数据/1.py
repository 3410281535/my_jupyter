import pandas as pd

# 读取 Excel 文件

df = pd.read_excel("社会融资规模.xls", header=None, names=['date', 'value'])

# 删除日期列
df.drop(columns=['date'], inplace=True)

# 初始化季度标识和合并后的数据列表
quarters = []
merged_values = []

# 获取数据的行数
num_rows = len(df)

# 生成季度标识和合并数据
quarter_counter = 3
year_counter = 2002

for i in range(0, num_rows, 3):
	if i + 2 < num_rows:
		# 合并数据
		merged_value = df.loc[i, 'value'] + df.loc[i + 1, 'value'] + df.loc[i + 2, 'value']

		# 添加到列表中
		quarters.append(f"{year_counter}-Q{quarter_counter}")
		merged_values.append(merged_value)

		# 更新季度标识
		if quarter_counter == 12:
			quarter_counter = 3
			year_counter += 1
		else:
			quarter_counter += 3
	elif i + 1 == num_rows:
		# 如果最后只剩两行
		merged_value = df.loc[i, 'value'] + df.loc[i + 1, 'value']
		quarters.append(f"{year_counter}-Q{quarter_counter}")
		merged_values.append(merged_value)
	elif i == num_rows:
		# 如果最后只剩一行
		merged_value = df.loc[i, 'value']
		quarters.append(f"{year_counter}-Q{quarter_counter}")
		merged_values.append(merged_value)

# 创建新的DataFrame
quarterly_df = pd.DataFrame({
	'quarter': quarters,
	'total': merged_values
})

# 打印结果
print(quarterly_df)

# 保存到新的Excel文件
output_file = '社会融资规模季度数据.xlsx'
quarterly_df.to_excel(output_file, index=False)
print(f"季度数据已保存到 {output_file}")