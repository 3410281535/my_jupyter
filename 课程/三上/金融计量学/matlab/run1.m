% 初始化环境
clearvars -except
clc

%% 导入工具箱
toolboxPaths = {'E:\jupyter_program\课程\金融计量学\matlab\VAR_Toolbox_2.0\VAR', 
               'E:\jupyter_program\课程\金融计量学\matlab\VAR_Toolbox_2.0\Utils',
               'E:\jupyter_program\课程\金融计量学\matlab\VAR_Toolbox_2.0\Stats',
               'E:\jupyter_program\课程\金融计量学\matlab\VAR_Toolbox_2.0\Auxiliary'};
for i = 1:length(toolboxPaths)
    addpath(toolboxPaths{i});
end

%% 数据处理
load('work.txt');
环比增长率 = 100 * diff(log('work')); % 环比增长率转换

%% 参数设置
变量标签 = {'GDP Deflator Growth Rate', 'Real GDP Growth Rate', 'M2 Growth Rate'};
图表标题 = {'GDP Deflator Time Series', 'Real GDP Growth Rate Time Series', 'M2 Time Series'};
颜色数组 = {'b', 'g', 'r'};
时间向量 = 1992:0.25:2024;

%% 数据可视化
figure;
for i = 1:3
    subplot(3,1,i);
    plot(time向量, 环比增长率(:,i), 'Color', 颜色数组{i}, 'LineWidth', 2);
    xlabel('Time');
    ylabel(变量标签{i});
    title(图表标题{i});
    grid on;
end

%% VAR模型估计
滞后阶数 = 1;
包含截距 = 1;
[模型, 模型选项] = VARmodel(环比增长率, 滞后阶数, 包含截距);

%% 短期约束识别
模型选项.ident = 'oir';
模型选项.nsteps = 20;

% 脉冲响应函数
[响应函数, 模型] = VARir(模型, 模型选项);

% 置信区间
[置信下界，置信上界，中位数，置信条] = VARirband(模型, 模型选项);

%% 脉冲响应分析
figure;
for i = 1:3
    for j = 1:3
        subplot(3, 3, (i - 1) * 3 + j);
        plot(1:20, 中位数(:, j, i), 'Color', 'r', 'LineWidth', 3);
        hold on;
        plot(1:20, 置信下界(:, j, i), 'Color', 'b', 'LineWidth', 1, 'LineStyle', '--');
        plot(1:20, 置信上界(:, j, i), 'Color', 'b', 'LineWidth', 1, 'LineStyle', '--');
        plot(1:20, zeros(20, 1), 'Color', 'k', 'LineWidth', 1, 'LineStyle', ':');
        hold off;
        
        % 设置标题和标签
        if i == 1
            title('Supply Shock', 'FontSize', 14);
        elseif i == 2
            title('Demand Shock', 'FontSize', 14);
        else
            title('Monetary Policy Shock', 'FontSize', 14);
        end
        ylabel(变量标签{j}, 'FontSize', 14);
        xlabel('Steps', 'FontSize', 14);
        set(gca, 'FontSize', 12, 'Color', 'none');
        grid on;
    end
end

%% 预测误差方差分解
[方差分解， 模型] = VARfevd(模型, 模型选项);
figure;
for i = 1:3
    for j = 1:3
        subplot(3, 3, (i - 1) * 3 + j);
        plot(1:20, 方差分解(:, j, i) * 1000, 'Color', 颜色数组{i}, 'LineWidth', 2);
        hold on;
        plot(1:20, zeros(20, 1), 'Color', 'k', 'LineWidth', 1, 'LineStyle', ':');
        hold off;
        
        % 设置标题和标签
        if i == 1
            title('Supply Shock FEVD', 'FontSize', 14);
        elseif i == 2
            title('Demand Shock FEVD', 'FontSize', 14);
        else
            title('Monetary Policy Shock FEVD', 'FontSize', 14);
        end
        ylabel(变量标签{j}, 'FontSize', 14);
        xlabel('Steps', 'FontSize', 14);
        ylim([0 1000]);
        set(gca, 'FontSize', 12);
        grid on;
    end
end

%% 历史分解
[响应函数， 模型] = VARir(模型, 模型选项);
历史分解 = VARhd(模型);

%% 供给冲击历史分解
供给冲击 = 历史分解.shock(:,:,1);
去趋势数据 = 环比增长率 - mean(环比增长率);

figure;
bar(time向量, 供给冲击, 'stacked');
hold on;
plot(time向量, 去趋势数据(:,1), 'Color', 'k', 'LineWidth', 2);
legend('Supply Shock', 'Demand Shock', 'Monetary Policy Shock', 'GDP Deflator');
xlabel('Time', 'FontSize', 18);
ylabel('Historical Decomposition', 'FontSize', 18);
axis tight;
set(gca, 'FontSize', 18, 'Color', 'none');
grid on;

%% 需求冲击历史分解
需求冲击 = 历史分解.shock(:,:,2);
去趋势数据 = 环比增长率 - mean(环比增长率);

figure;
bar(time向量, 需求冲击, 'stacked');
hold on;
plot(time向量, 去趋势数据(:,2), 'Color', 'k', 'LineWidth', 2);
legend('Supply Shock', 'Demand Shock', 'Monetary Policy Shock', 'Real GDP');
xlabel('Time', 'FontSize', 18);
ylabel('Historical Decomposition', 'FontSize', 18);
axis tight;
set(gca, 'FontSize', 18, 'Color', 'none');
grid on;

%% 货币政策冲击历史分解
货币政策冲击 = 历史分解.shock(:,:,3);
去趋势数据 = 环比增长率 - mean(环比增长率);

figure;
bar(time向量, 货币政策冲击, 'stacked');
hold on;
plot(time向量, 去趋势数据(:,3), 'Color', 'k', 'LineWidth', 2);
legend('Supply Shock', 'Demand Shock', 'Monetary Policy Shock', 'M2');
xlabel('Time', 'FontSize', 18);
ylabel('Historical Decomposition', 'FontSize', 18);
axis tight;
set(gca, 'FontSize', 18, 'Color', 'none');
grid on;