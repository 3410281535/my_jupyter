%% VAR(24) GDP和平减指数
clear
clc
tic

%% 加载工具箱
addpath('E:\jupyter_program\课程\金融计量学\matlab\VAR_Toolbox_2.0\VAR')
addpath('E:\jupyter_program\课程\金融计量学\matlab\VAR_Toolbox_2.0\Utils')
addpath('E:\jupyter_program\课程\金融计量学\matlab\VAR_Toolbox_2.0\Stats')
addpath('E:\jupyter_program\课程\金融计量学\matlab\VAR_Toolbox_2.0\Auxiliary')

%% 读取数据
load GDP_DEF.txt %1992Q1-2024Q2

data = 100 * diff(log(GDP_DEF)); % 转换为环比增长率

%% 简化式VAR模型估计
nlag = 1;%滞后阶数
const = 1;%截距项

[VAR, VARopt] = VARmodel(data,nlag,const); 

%% 长期零约束识别
VARopt.ident = 'bq'; %长期零约束
VARopt.nsteps = 20;
[IRF, VAR] = VARir(VAR,VARopt);

VARopt.ndraws = 1000
VARopt.pctg = 90
[INF,SUP,MED,BAR] = VARirband(VAR,VARopt);%INF脉冲响应上界

%% 脉冲响应分析带置信区间
VARopt.ndraws = 1000;
VARopt.pctg = 90;
[INF, SUP, MED, BAR] = VARirband(VAR, VARopt); % 估计脉冲响应的置信区间（上界INF，下界SUP，中位数MED）

figure
for ii = 1:size(data, 2) % 对每个冲击变量
    for jj = 1:size(data, 2) % 对每个响应变量
        subplot(2, 2, (ii-1)*2+jj) % 由于有两个变量，所以是2行2列的子图布局
        plot(1:VARopt.nsteps, MED(:, jj, ii), "Color", 'r', 'LineWidth', 3, 'LineStyle', '-') % 脉冲响应中位数
        hold on
        plot(1:VARopt.nsteps, INF(:, jj, ii), "Color", 'b', 'LineWidth', 1, 'LineStyle', '--') % 脉冲响应90%置信区间的上界
        plot(1:VARopt.nsteps, SUP(:, jj, ii), "Color", 'b', 'LineWidth', 1, 'LineStyle', '--') % 脉冲响应90%置信区间的下界
        plot(1:VARopt.nsteps, zeros(VARopt.nsteps, 1), "Color", 'k', 'LineWidth', 1, 'LineStyle', ':') % x轴零线
        hold off
        if ii == 1
            xlabel('期数')
            if jj == 1
                title('GDP冲击对GDP的影响')
                ylabel('GDP环比增长率变化')
            else
                title('GDP冲击对平减指数的影响')
                ylabel('平减指数环比增长率变化')
            end
        elseif ii == 2
            xlabel('期数')
            if jj == 1
                title('平减指数冲击对GDP的影响')
                ylabel('GDP环比增长率变化')
            else
                title('平减指数冲击对平减指数的影响')
                ylabel('平减指数环比增长率变化')
            end
        end
        set(gca, 'FontSize', 20, 'Color', 'none', 'Box', 'off')
        grid on
    end
end
toc