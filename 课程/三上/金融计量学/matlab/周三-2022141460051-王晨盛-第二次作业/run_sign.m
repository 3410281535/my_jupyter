%% VAR(1) 三变量宏观模型：通胀、GDP、M2
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
ylabels = {'GDP平减指数', '实际GDP', 'M2总量'};
% 定义标题
titles = {'GDP平减指数时间序列图', '实际GDP环比增长率时间序列图', 'M2总量环比增长率时间序列图'};
title_shock = {'供给冲击', '需求冲击', '货币政策冲击'}; 
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

%% 简化式VAR模型估计
nlag  = 1; % 滞后阶数
const = 1; % 截距

[VAR, VARopt] = VARmodel(data,nlag,const);

%% 符号约束的识别

% 假设：
% 1.供给冲击：通胀对供给冲击在            1-4的响应为正
%              
%             M2总量对供给冲击在          1-4的响应为正
%
% 2.需求冲击：
%             
%
%             
% 3. 货币政策冲击：通胀对货币政策冲击在   2-4的响应为正
%                  实际GDP对货币政策冲击在1-4的响应为正
%                  M2总量对货币政策冲击在 1-4的响应为正



% 供给冲击
R(:,:,1) = [1, 4,  1;
            0, 0,  0;
            1, 4,  1 ];
% 需求冲击
R(:,:,2) = [0, 0,  0;
            0, 0,  0;
            0, 0,  0 ];
% 货币政策冲击
R(:,:,3) = [2, 4,  1; % 不能是141
            1, 4,  1;
            1, 4,  1 ];
        
VARopt.nsteps = 20; % 设置预测步数为 20 个周期
VARopt.ndraws = 1000; % 设置抽样次数为 1000 次

% 进行符号约束识别并计算 IRFs, FEVD, HD
SRout = SR(VAR,R,VARopt);

%% 脉冲响应分析
figure
for ii = 1:3 % 变量序号&行序
    for jj = 1:3 % % 冲击序号&列序
        subplot(3, 3, (ii - 1) * 3 + jj)
        plot(1:20,SRout.IRFmed(:,ii,jj),"Color",colors{ii},"LineWidth",3,"LineStyle","-")
        hold on
        plot(1:20,SRout.IRFinf(:,ii,jj),"Color",colors{ii},"LineWidth",1,"LineStyle","--")
        hold on
        plot(1:20,SRout.IRFsup(:,ii,jj),"Color",colors{ii},"LineWidth",1,"LineStyle","--")
        hold on
        plot(1:20,zeros(20,1),"Color",'k','LineWidth',1,'LineStyle',':')
        title(title_shock{ii})
        ylabel(ylabels{jj})          
        axis tight
        set(gca,'FontSize',18,'Color','none')
        grid on
        
    end
end

%% 方差分解分析
figure
for ii = 1:3 % 变量序号&行序
    for jj = 1:3 % 只有一个冲击（货币政策冲击）
        subplot(3, 3, (ii - 1) * 3 + jj)
        plot(1:20,SRout.FEVDmed(:,ii,jj),"Color",colors{ii},"LineWidth",3,"LineStyle","-")
        hold on
        plot(1:20,SRout.FEVDinf(:,ii,jj),"Color",colors{ii},"LineWidth",1,"LineStyle","--")
        hold on
        plot(1:20,SRout.FEVDsup(:,ii,jj),"Color",colors{ii},"LineWidth",1,"LineStyle","--")
        hold on
        plot(1:20,zeros(20,1),"Color",'k','LineWidth',1,'LineStyle',':')

        title(title_shock{ii})
        ylabel(ylabels{jj})    
        ylim([0 1])
        xlim([1 20])
        axis tight
        set(gca,'FontSize',18,'Color','none')
        grid on
        ylim([0 1])
    end
end

%% 历史分解
GDPDE_HD = squeeze(SRout.HDmed(:,1,:));
REGDP_HD = squeeze(SRout.HDmed(:,2,:));
M2_HD = squeeze(SRout.HDmed(:,3,:));
data_demean = data - mean(data);

% GDP平减指数
figure 
bar(tim,GDPDE_HD,'stacked')
hold on
plot(tim,data_demean(:,1),"Color",'k','LineWidth',2,'LineStyle','-')

legend('需求冲击','供给冲击', '货币政策冲击','GDP平减指数')
legend('boxoff')
axis tight
set(gca,'FontSize',18,'Color','none')
grid on

% 实际GDP
figure 
bar(tim,REGDP_HD,'stacked')
hold on
plot(tim,data_demean(:,2),"Color",'k','LineWidth',2,'LineStyle','-')

legend('需求冲击','供给冲击', '货币政策冲击','实际GDP')
legend('boxoff')
axis tight
set(gca,'FontSize',18,'Color','none')
grid on

% M2总量
figure 
bar(tim,M2_HD,'stacked')
hold on
plot(tim,data_demean(:,3),"Color",'k','LineWidth',2,'LineStyle','-')

legend('需求冲击','供给冲击', '货币政策冲击','M2总量')
legend('boxoff')
axis tight
set(gca,'FontSize',18,'Color','none')
grid on
%close all
toc