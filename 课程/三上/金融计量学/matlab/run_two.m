%% VAR(1)
clear
clc
tic
%% 加载工具箱
addpath('E:\jupyter_program\课程\金融计量学\matlab\VAR_Toolbox_2.0\VAR')
addpath('E:\jupyter_program\课程\金融计量学\matlab\VAR_Toolbox_2.0\Utils')
addpath('E:\jupyter_program\课程\金融计量学\matlab\VAR_Toolbox_2.0\Stats')
addpath('E:\jupyter_program\课程\金融计量学\matlab\VAR_Toolbox_2.0\Auxiliary')

%% 读取数据
load Two.txt

%% 数据图
figure
tim = 1996:0.25:2024.50;%1996Q1-2024Q3
subplot(1, 2, 1)
plot(tim, Two(:,1), "Color",'r', 'LineWidth', 3, 'LineStyle', '-')
xlabel('时间')
ylabel('%')
title('中国GDP同比增速')
axis tight
set(gca, 'FontSize', 20, 'Color', 'none')
grid on

subplot(1, 2, 2)
plot(tim, Two(:,2), "Color",'r', 'LineWidth', 3, 'LineStyle', '-')
xlabel('时间')
ylabel('%')
title('同业拆借利率')
axis tight
set(gca, 'FontSize', 20, 'Color', 'none')
grid on

%% 简化式VAR估计
nlag = 1;%滞后阶数
const = 1;%有截距项

[VAR, VARopt] = VARmodel(Two,nlag,const); 

%短期零约束识别
VARopt.ident = 'oir';
VARopt.nsteps = 40;
[IRF, VAR] = VARir(VAR,VARopt);

%%脉冲响应分析
%%货币政策冲击对GDP的影响
figure
plot(1:40,IRF(:,1,2), 'Color','b','LineWidth',3,'LineStyle', '-')
xlabel('响应期数')
ylabel('%')
axis tight
set(gca, 'FontSize', 20, 'Color', 'none')
grid


%%分样本 1996Q1-2009Q4, 2010Q1-2024Q3
nlag = 1;
const = 1;

Two = Two(1:end,:);

data1 = Two(1:56,:);%1996-2009
data2 = Two(57:end,:);%2010-2024
data3 = Two;%2010-2024

[VAR1, VARopt1] = VARmodel(data1, nlag, const);
[VAR2, VARopt2] = VARmodel(data2, nlag, const);
[VAR3, VARopt3] = VARmodel(data3, nlag, const);
%% 短期零约束识别
VARopt1.indent = 'oir';
VARopt1.nsteps = 40;

VARopt2.indent = 'oir';
VARopt2.nsteps = 40;

VARopt3.indent = 'oir';
VARopt3.nsteps = 40;

[IRF1, VAR1] = VARir(VAR1, VARopt1);
[IRF2, VAR2] = VARir(VAR2, VARopt2);
[IRF3, VAR3] = VARir(VAR3, VARopt3);

%%
figure
plot(1:40,IRF1(:,1,2),'Color','b','LineWidth',3,'LineStyle','-')
hold on
plot(1:40,IRF2(:,1,2),'Color','r','LineWidth',3,'LineStyle','-')
hold on
plot(1:40,IRF3(:,1,2),'Color','k','LineWidth',3,'LineStyle','-')

legend('1996-2009','2010-2024','1996-2024','FontSize',20)
xlabel('响应期数')
ylabel('%')
title('货币政策冲击对GDP的影响')
axis tight
set(gca, 'FontSize', 20, 'Color', 'none')
grid on

%%移动窗口分析
GDP_to_MP = zeros(40,60);
 for ii = 1:60
    data = Two(1:ii + 55,:); 
    [VAR, VARopt] = VARmodel(data, nlag, const);
    
    [IRF, VAR] = VARir(VAR, VARopt);
    GDP_to_MP_temp = IRF(:,1,2);
    GDP_to_MP(:,ii) = GDP_to_MP_temp;
    ii
 end
 
%% 三维脉冲响应
[X,Y] = meshgrid(2009.75:0.25:2024.50,1:40);
figure

surf(X,Y,GDP_to_MP)
%axis tight
set(gca, 'FontSize', 20, 'Color', 'none')
grid on

clear
clc
%% 简化式VAR估计
nlag = 1;%滞后阶数
const = 1;%有截距项

load Two.txt
Two = Two(1:end - 19,:);

[VAR,VARopt] = VARmodel(Two,nlag,const);

%%置信区间分析
VARopt.ndraws = 1000;%设置抽样次数
VARopt.pctg = 68;%设置置信区间,68为最小置信带
VARopt.indent = 'oir';
VARopt.nsteps = 40;

[INF,SUP,MED,BAR] = VARirband(VAR,VARopt);%INF脉冲响应上界
%%
figure
plot(1:40,MED(:,1,2),'Color','k','LineWidth',3,'LineStyle','-')
hold on
plot(1:40,INF(:,1,2),'Color','k','LineWidth',2,'LineStyle','--')
hold on
plot(1:40,SUP(:,1,2),'Color','k','LineWidth',2,'LineStyle','--')
hold on
plot(1:40,zeros(40,1),'Color','k','LineWidth',1,'LineStyle',':')
%legend('1996-2009','2010-2024','1996-2024','FontSize',20)
xlabel('响应期数')
ylabel('%')
title('货币政策冲击对GDP的影响')
axis tight
set(gca, 'FontSize', 20, 'Color', 'none')
grid on
%VARopt.pctg = 90;时不显著 

%%方差分解
[FEVD, VAR] = VARfevd(VAR,VARopt);

%%方差分解分析
figure
plot(1:40,FEVD(:,1,2),'Color','k','LineWidth',2,'LineStyle','-')

xlabel('响应期数')
xlim([1,40])
ylim([0, 1])
title('货币政策冲击对GDP的解释比重')

set(gca, 'FontSize', 20, 'Color', 'none')
grid on

%%方差分解分析
figure
plot(1:40,FEVD(:,1,1),'Color','k','LineWidth',2,'LineStyle','-')

xlabel('响应期数')
xlim([1,40])
ylim([0, 1])
title('货币政策冲击对利率的解释比重')
%axis tight
set(gca, 'FontSize', 20, 'Color', 'none')
grid on

toc;
 