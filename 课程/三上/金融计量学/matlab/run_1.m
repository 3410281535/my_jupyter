% �����ļ�����·��
jumpFile = 'һ���ڴ������.txt';
continuousFile = '����ͬҵ�������.txt';
quarterlyFile = 'M2.txt';

% �������ڷ�Χ
startDate = '1996-08-23'; % �����ļ��е���������
endDate = '2024-09-30';   % ��������

% ��ȡ��Ծʽ�����ļ�
jumpTable = readtable(jumpFile, 'Delimiter', '\t');
jumpDates = jumpTable.���; % ���������е�����Ϊ '���'
jumpValues = jumpTable.��ֵ; % ������ֵ�е�����Ϊ '��ֵ'

% ��ȡ����ʽ�����ļ�
continuousTable = readtable(continuousFile, 'Delimiter', '\t');
continuousDates = continuousTable.����; % ���������е�����Ϊ '����'
continuousValues = continuousTable.��ֵ; % ������ֵ�е�����Ϊ '��ֵ'

% ��ȡ���������ļ�
quarterlyTable = readtable(quarterlyFile, 'Delimiter', '\t');
quarterlyDates = quarterlyTable.����; % ���������е�����Ϊ '����'
quarterlyValues = quarterlyTable.��ֵ; % ������ֵ�е�����Ϊ '��ֵ'

% ת����Ծʽ���ں�����ʽ����Ϊ������
jumpDates_num = datenum(jumpDates, 'yyyy-mm-dd');
continuousDates_num = datenum(continuousDates, 'yyyy-mm-dd');

% ת����������Ϊ���ȵĵ�һ�죨 yyyy-mm-01 ��
quarterlyDates_converted = cellfun(@(x) sprintf('%s-01', x), quarterlyDates, 'UniformOutput', false);
quarterlyDates_num = datenum(quarterlyDates_converted, 'yyyy-mm-dd');

% ����������ȷ��ֻ���������ڷ�Χ�ڵ�����
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

% ��ͼ
figure;
hold on;

% ��Ծʽ��������
plot(jumpDates_num_filtered, jumpValues_filtered, '-o', 'DisplayName', 'һ���ڴ������');

% ����ʽ��������
plot(continuousDates_num_filtered, continuousValues_filtered, '-s', 'DisplayName', '����ͬҵ�������');

% ������������
plot(quarterlyDates_num_filtered, quarterlyValues_filtered, '-x', 'DisplayName', 'M2');

% ����ͱ�ǩ
title('��ͬ���ڸ�ʽ�����ݱȽ�');
xlabel('����');
ylabel('��ֵ');
datetick('x', 'yyyy-mm');

% ͼ��
legend;

% ���������᷶Χ
xlim([startDate_num, endDate_num]);

hold off;