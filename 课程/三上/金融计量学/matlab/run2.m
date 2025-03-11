% 初始化环境
clear
clc
tic

%% 导入工具箱
varPath = 'E:\jupyter_program\课程\金融计量学\matlab\VAR_Toolbox_2.0';
addpath([varPath '\VAR']);
addpath([varPath '\Utils']);
addpath([varPath '\Stats']);
addpath([varPath '\Auxiliary']);

%% 读取数据
load('work.txt');
economicData = 100 * diff(log(work)); % 转换为环比增长率

%% 定义标签和标题
ylabelGrowth = {'GDP平减指数环比增长率', '实际GDP环比增长率', 'M2总量环比增长率'};
ylabelTotal = {'GDP平减指数', '实际GDP', 'M2总量'};
titlesTimeSeries = {'GDP平减指数时间序列图', '实际GDP环比增长率时间序列图', 'M2总量环比增长率时间序列图'};
titlesShock = {'供给冲击', '需求冲击', '货币政策冲击'};
colorOptions = {'b', 'g', 'r'};
timeVector = 1992:0.25:2024;

%% 数据可视化
figure;
for i = 1:3
    subplot(3,1,i);
    plot(timeVector, economicData(:,i), 'Color', colorOptions{i}, 'LineWidth', 2);
    xlabel('时间');
    ylabel(ylabelGrowth{i});
    title(titlesTimeSeries{i});
    grid on;
end

%% VAR(1)模型估计
lags = 1; % 滞后阶数
interceptFlag = 1; % 截距项

[varModel, varOptions] = VARmodel(economicData, lags, interceptFlag);

%% 符号约束识别
responseConstraints = zeros(3, 4, 3);
responseConstraints(:,:,1) = [1, 4, 1; 0, 0, 0; 1, 4, 1]; % 供给冲击
responseConstraints(:,:,2) = [0, 0, 0; 0, 0, 0; 0, 0, 0]; % 需求冲击
responseConstraints(:,:,3) = [2, 4, 1; 1, 4, 1; 1, 4, 1]; % 货币政策冲击

varOptions.nsteps = 20; % 预测步数
varOptions.ndraws = 1000; % 抽样次数

% 进行符号约束识别并计算 IRFs, FEVD, HD
[shockResponse] = SR(varModel, responseConstraints, varOptions);

%% 脉冲响应分析
figure;
for i = 1:3
    for j = 1:3
        subplot(3, 3, (i - 1) * 3 + j);
        plot(1:20, shockResponse.IRFmed(:,i,j), "Color", colorOptions{i}, "LineWidth", 3, "LineStyle", "-");
        hold on;
        plot(1:20, shockResponse.IRFinf(:,i,j), "Color", colorOptions{i}, "LineWidth", 1, "LineStyle", "--");
        plot(1:20, shockResponse.IRFsup(:,i,j), "Color", colorOptions{i}, "LineWidth", 1, "LineStyle", "--");
        plot(1:20, zeros(20,1), "Color", 'k', "LineWidth", 1, "LineStyle", ":");
        hold off;
        title(titlesShock{j});
        ylabel(ylabelTotal{i});
        axis tight;
        set(gca, 'FontSize', 18, 'Color', 'none');
        grid on;
    end
end

%% 方差分解分析
figure;
for i = 1:3
    for j = 1:3
        subplot(3, 3, (i - 1) * 3 + j);
        plot(1:20, shockResponse.FEVDmed(:,i,j), "Color", colorOptions{i}, "LineWidth", 3, "LineStyle", "-");
        hold on;
        plot(1:20, shockResponse.FEVDinf(:,i,j), "Color", colorOptions{i}, "LineWidth", 1, "LineStyle", "--");
        plot(1:20, shockResponse.FEVDsup(:,i,j), "Color", colorOptions{i}, "LineWidth", 1, "LineStyle", "--");
        plot(1:20, zeros(20,1), "Color", 'k', "LineWidth", 1, "LineStyle", ":");
        hold off;
        title(titlesShock{j});
        ylabel(ylabelTotal{i});
        ylim([0 1]);
        xlim([1 20]);
        axis tight;
        set(gca, 'FontSize', 18, 'Color', 'none');
        grid on;
    end
end

%% 历史分解
HistoricalDecompositionGDPDE = squeeze(shockResponse.HDmed(:,1,:));
HistoricalDecompositionREGDP = squeeze(shockResponse.HDmed(:,2,:));
HistoricalDecompositionM2 = squeeze(shockResponse.HDmed(:,3,:));
detrendedData = economicData - mean(economicData);

% GDP平减指数历史分解
figure;
bar(timeVector, HistoricalDecompositionGDPDE, 'stacked');
hold on;
plot(timeVector, detrendedData(:,1), "Color", 'k', "LineWidth", 2, "LineStyle", "-");
legend('需求冲击', '供给冲击', '货币政策冲击', 'GDP平减指数');
legend('boxoff');
axis tight;
set(gca, 'FontSize', 18, 'Color', 'none');
grid on;

% 实际GDP历史分解
figure;
bar(timeVector, HistoricalDecompositionREGDP, 'stacked');
hold on;
plot(timeVector, detrendedData(:,2), "Color", 'k', "LineWidth", 2, "LineStyle", "-");
legend('需求冲击', '供给冲击', '货币政策冲击', '实际GDP');
legend('boxoff');
axis tight;
set(gca, 'FontSize', 18, 'Color', 'none');
grid on;

% M2总量历史分解
figure;
bar(timeVector, HistoricalDecompositionM2, 'stacked');
hold on;
plot(timeVector, detrendedData(:,3), "Color", 'k', "LineWidth", 2, "LineStyle", "-");
legend('需求冲击', '供给冲击', '货币政策冲击', 'M2总量');
legend('boxoff');
axis tight;
set(gca, 'FontSize', 18, 'Color', 'none');
grid on;

%toc % 计时结束