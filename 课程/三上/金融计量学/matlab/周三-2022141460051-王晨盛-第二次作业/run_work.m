%% VAR(12) 三变量宏观模型：通胀、GDP、M2
clear
clc
tic
 
%% 加载工具箱
addpath('E:\jupyter_program\课程\金融计量学\matlab\VAR_Toolbox_2.0\VAR')
addpath('E:\jupyter_program\课程\金融计量学\matlab\VAR_Toolbox_2.0\Utils')
addpath('E:\jupyter_program\课程\金融计量学\matlab\VAR_Toolbox_2.0\Stats')
addpath('E:\jupyter_program\课程\金融计量学\matlab\VAR_Toolbox_2.0\Auxiliary')

%% 读取数据
load work.txt
data = 100 * diff(log(work)); % 转换为环比增长率

%%
% 定义y轴标签
yLabels = {'GDP平减指数环比增长率', '实际GDP环比增长率', 'M2总量环比增长率'};
% 定义标题
titles = {'GDP平减指数时间序列图', '实际GDP环比增长率时间序列图', 'M2总量环比增长率时间序列图'};
% 颜色数组
colors = {'b', 'g', 'r'};
tim = 1992:0.25:2024.00;


% 数据可视化
figure 
for ii = 1:3
   subplot(3,1,ii);
    plot(tim, data(:,ii), "Color", colors{ii}, "LineWidth", 2, "LineStyle", "-");
    xlabel('时间');
    ylabel(yLabels{ii});
    title(titles{ii});
    grid on;
end
   
%% 简化式VAR估计
nlag = 1; % 滞后阶数
const = 1; % 有截距项

[VAR, VARopt] = VARmodel(data, nlag, const); 

%% 短期零约束识别
VARopt.ident = 'oir';
VARopt.nsteps = 20;

% 脉冲响应函数
[IRF, VAR] = VARir(VAR, VARopt);

VARopt.ndraws = 1000; % 抽样次数
VARopt.pctg = 90; % 置信区间

% 计算脉冲响应函数的置信区间
[INF, SUP, MED, BAR] = VARirband(VAR, VARopt);

%% 脉冲响应分析
figure
for ii = 1:3 % 
    for jj = 1:3 % 变量序号：通胀、GDP、M2
        subplot(3, 3, (ii - 1) * 3 + jj)
        plot(1:20, MED(:, jj, ii), "Color", 'r', 'LineWidth', 3, 'LineStyle', '-'); % 中位数
        hold on
        plot(1:20, INF(:, jj, ii), "Color", 'b', 'LineWidth', 1, 'LineStyle', '--'); % 下界
        plot(1:20, SUP(:, jj, ii), "Color", 'b', 'LineWidth', 1, 'LineStyle', '--'); % 上界
        plot(1:20, zeros(20, 1), "Color", 'k', 'LineWidth', 1, 'LineStyle', ':'); % x轴
        hold off
        
        % 设置标题
        if ii == 1
            title('供给冲击', 'FontSize', 14)
        elseif ii == 2
            title('需求冲击', 'FontSize', 14)
        else
            title('货币政策冲击', 'FontSize', 14)
        end
        
        % 设置y轴标签
        ylabel(yLabels{jj}, 'FontSize', 14)
        
        % 设置x轴标签
        xlabel('时间步数', 'FontSize', 14)
        
        % 调整图形属性
        set(gca, 'FontSize', 12, 'Color', 'none')
        grid on
    end
end   

%% 预测误差方差分解
[FEVD, VAR] = VARfevd(VAR, VARopt);
figure
for ii = 1:3 % 冲击序号：供给冲击、需求冲击、货币政策冲击
    for jj = 1:3 % 变量序号
        subplot(3, 3, (ii - 1) * 3 + jj)
        plot(1:20, FEVD(:, jj, ii) * 1000, "Color", colors{ii}, "LineWidth", 2, "LineStyle", "-"); % 乘以100便于展示百分比
        hold on
        plot(1:20, zeros(20, 1), "Color", 'k', 'LineWidth', 1, 'LineStyle', ':') % x轴
        hold off
        
        % 设置标题
        if ii == 1
            title('供给冲击对各变量的FEVD', 'FontSize', 14)
        elseif ii == 2
            title('需求冲击对各变量的FEVD', 'FontSize', 14)
        else
            title('货币政策冲击对各变量的FEVD', 'FontSize', 14)
        end
        
        % 设置y轴标签
        ylabel(yLabels{jj}, 'FontSize', 14)
        
        % 设置x轴标签
        xlabel('时间步数', 'FontSize', 14)
        
        % 设置y轴范围
        ylim([0 1000]); % 百分比形式
        
        % 调整图形属性
        set(gca, 'FontSize', 12)
        grid on
    end
end

%% 历史分解
[IRF, VAR] = VARir(VAR, VARopt);
HD = VARhd(VAR);

%% 供给冲击
work_HD = HD.shock(:,:,1);

data_demean = data - mean(data);

figure
bar(tim, work_HD, 'stacked')
hold on
plot(tim, data_demean(:,1), "Color", 'k', "LineWidth", 2, "LineStyle", "-")

% 图例说明
legend('供给冲击', '需求冲击', '货币政策冲击', 'GDP平减指数')

% 设置x轴标签
xlabel('时间', 'FontSize', 18)

% 设置y轴标签
ylabel('历史分解贡献', 'FontSize', 18)

axis tight
set(gca, 'FontSize', 18, 'Color', 'none')
grid on

%% 需求冲击
work_HD = HD.shock(:,:,2);

data_demean = data - mean(data);

figure
bar(tim, work_HD, 'stacked')
hold on
plot(tim, data_demean(:,2), "Color", 'k', "LineWidth", 2, "LineStyle", "-")

% 图例说明
legend('供给冲击', '需求冲击', '货币政策冲击', '实际GDP')

% 设置x轴标签
xlabel('时间', 'FontSize', 18)

% 设置y轴标签
ylabel('历史分解贡献', 'FontSize', 18)

axis tight
set(gca, 'FontSize', 18, 'Color', 'none')
grid on

%% 货币政策冲击
work_HD = HD.shock(:,:,3);

data_demean = data - mean(data);

figure
bar(tim, work_HD, 'stacked')
hold on
plot(tim, data_demean(:,3), "Color", 'k', "LineWidth", 2, "LineStyle", "-")

% 图例说明
legend('供给冲击', '需求冲击', '货币政策冲击', 'M2总量')

% 设置x轴标签
xlabel('时间', 'FontSize', 18)

% 设置y轴标签
ylabel('历史分解贡献', 'FontSize', 18)

axis tight
set(gca, 'FontSize', 18, 'Color', 'none')
grid on