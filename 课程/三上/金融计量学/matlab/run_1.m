% 定义文件名和路径
jumpFile = '一年期存款利率.txt';
continuousFile = '七天同业拆借利率.txt';
quarterlyFile = 'M2.txt';

% 定义日期范围
startDate = '1996-08-23'; % 三个文件中的最早日期
endDate = '2024-09-30';   % 最晚日期

% 读取跳跃式日期文件
jumpTable = readtable(jumpFile, 'Delimiter', '\t');
jumpDates = jumpTable.年份; % 假设日期列的名称为 '年份'
jumpValues = jumpTable.数值; % 假设数值列的名称为 '数值'

% 读取连续式日期文件
continuousTable = readtable(continuousFile, 'Delimiter', '\t');
continuousDates = continuousTable.日期; % 假设日期列的名称为 '日期'
continuousValues = continuousTable.数值; % 假设数值列的名称为 '数值'

% 读取季度日期文件
quarterlyTable = readtable(quarterlyFile, 'Delimiter', '\t');
quarterlyDates = quarterlyTable.日期; % 假设日期列的名称为 '日期'
quarterlyValues = quarterlyTable.数值; % 假设数值列的名称为 '数值'

% 转换跳跃式日期和连续式日期为日期数
jumpDates_num = datenum(jumpDates, 'yyyy-mm-dd');
continuousDates_num = datenum(continuousDates, 'yyyy-mm-dd');

% 转换季度日期为季度的第一天（ yyyy-mm-01 ）
quarterlyDates_converted = cellfun(@(x) sprintf('%s-01', x), quarterlyDates, 'UniformOutput', false);
quarterlyDates_num = datenum(quarterlyDates_converted, 'yyyy-mm-dd');

% 过滤数据以确保只绘制在日期范围内的数据
startDate_num = datenum(startDate, 'yyyy-mm-dd');
endDate_num = datenum(endDate, 'yyyy-mm-dd');

jump_ind = jumpDates_num >= startDate_num & jumpDates_num <= endDate_num;
continuous_ind = continuousDates_num >= startDate_num & continuousDates_num <= endDate_num;
quarterly_ind = quarterlyDates_num >= startDate_num & quarterlyDates_num <= endDate_num;

jumpDates_filtered = jumpDates(jump_ind);
jumpValues_filtered = jumpValues(jump_ind);
jumpDates_num_filtered = jumpDates_num(jump_ind);

continuousDates_filtered = continuousDates(continuous_ind);
continuousValues_filtered = continuousValues(continuous_ind);
continuousDates_num_filtered = continuousDates_num(continuous_ind);

quarterlyDates_filtered = quarterlyDates_converted(quarterly_ind);
quarterlyValues_filtered = quarterlyValues(quarterly_ind);
quarterlyDates_num_filtered = quarterlyDates_num(quarterly_ind);

% 绘图
figure;
hold on;

% 跳跃式日期数据
plot(jumpDates_num_filtered, jumpValues_filtered, '-o', 'DisplayName', '一年期存款利率');

% 连续式日期数据
plot(continuousDates_num_filtered, continuousValues_filtered, '-s', 'DisplayName', '七天同业拆借利率');

% 季度日期数据
plot(quarterlyDates_num_filtered, quarterlyValues_filtered, '-x', 'DisplayName', 'M2');

% 标题和标签
title('不同日期格式的数据比较');
xlabel('日期');
ylabel('数值');
datetick('x', 'yyyy-mm');

% 图例
legend;

% 调整日期轴范围
xlim([startDate_num, endDate_num]);

hold off;