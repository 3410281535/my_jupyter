%% VAR(24) 石油产量、世界产出和实际油价三变量模型
clear
clc
tic
%% 加载工具箱
addpath('E:\jupyter_program\课程\金融计量学\matlab\VAR_Toolbox_2.0\VAR')
addpath('E:\jupyter_program\课程\金融计量学\matlab\VAR_Toolbox_2.0\Utils')
addpath('E:\jupyter_program\课程\金融计量学\matlab\VAR_Toolbox_2.0\Stats')
addpath('E:\jupyter_program\课程\金融计量学\matlab\VAR_Toolbox_2.0\Auxiliary')

%% 读取数据
load data_oil.txt

%% 数据图
figure
for ii = 1:3
    subplot(3, 1, ii)
    tim = 1973+2/12:1/12:2007+12/12% 1996Q1-2024Q3
    plot(tim, data_oil(:,ii), "Color",'r', 'LineWidth', 3, 'LineStyle', '-')
    xlabel('时间')
    if ii == 1
        title('石油产量')
    elseif ii == 2
        title('世界产出')
    else
        title('实际油价')
    end
    axis tight
    set(gca, 'FontSize', 20, 'Color', 'none')
    grid on
end


%% 简化式VAR估计
nlag = 24;%滞后阶数
const = 1;%有截距项

[VAR, VARopt] = VARmodel(data_oil,nlag,const); 

%% 短期零约束识别
VARopt.ident = 'oir';
VARopt.nsteps = 20;
[IRF, VAR] = VARir(VAR,VARopt);

VARopt.ndraws = 1000;
VARopt.pctg = 90;
[INF,SUP,MED,BAR] = VARirband(VAR,VARopt);%INF脉冲响应上界
%% 脉冲响应分析
figure
for ii = 1:3 % 冲击序号：供给冲击、
    for jj = 1:3 % 变量序号
        subplot(3, 3, (ii - 1) * 3 + jj)
        plot(1:20, MED(:,jj,ii), "Color",'r', 'LineWidth', 3, 'LineStyle', '-')
        hold on
        plot(1:20, INF(:,jj,ii), "Color",'b', 'LineWidth', 1, 'LineStyle', '--')
        hold on
        plot(1:20, SUP(:,jj,ii), "Color",'b', 'LineWidth', 1, 'LineStyle', '--')
        hold on
        plot(1:20, zeros(20, 1), "Color",'k', 'LineWidth', 1, 'LineStyle', ':')
        if ii ==1
            title('石油供给冲击')
        elseif ii == 2
            title('总需求冲击')
        else
            title('专有型需求冲击')
        end
        
        if jj ==1
            ylabel('石油产量')
            ylim([-25, 15])
            xlim([1, 20])
        elseif jj == 2
            ylabel('世界产出')
            ylim([-5, 10])
            xlim([1, 20])
        else
            ylabel('实际油价')
            ylim([-8, 12])
            xlim([1, 20])
        end
        set(gca, 'FontSize', 18, 'Color', 'none')
        grid on
   end
end

%% 历史分解
[IRF, VAR] = VARir(VAR, VARopt);
HD = VARhd(VAR);
oil_HD = HD.shock(:,:,3);

data_demean = data_oil - mean(data_oil);

figure
bar(tim, oil_HD,'stacked')
hold on
plot(tim, data_demean(:,3),"Color",'k',"LineWidth",2,"LineStyle","-")

legend('石油供给冲击','总需求冲击','专有型需求冲击','实际油价')

axis tight
set(gca,'FontSize',18,'Color','none')
grid on

%% 预测误差方差分解
[FEVD, VAR] = VARfevd(VAR,VARopt);
