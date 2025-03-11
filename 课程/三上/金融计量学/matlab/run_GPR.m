%% VAR
clear
clc
tic
%% 加载工具箱
addpath('E:\jupyter_program\课程\金融计量学\matlab\VAR_Toolbox_2.0\VAR')
addpath('E:\jupyter_program\课程\金融计量学\matlab\VAR_Toolbox_2.0\Utils')
addpath('E:\jupyter_program\课程\金融计量学\matlab\VAR_Toolbox_2.0\Stats')
addpath('E:\jupyter_program\课程\金融计量学\matlab\VAR_Toolbox_2.0\Auxiliary')

%% 读取数据
load GPR.txt
%% 简化式VAR估计
nlag = 1;%滞后阶数
const = 1;%有截距项

[VAR, VARopt] = VARmodel(GPR,nlag,const); 
%% 短期零约束识别
VARopt.ident = 'oir';
VARopt.nsteps = 20;
[IRF, VAR] = VARir(VAR,VARopt);

VARopt.ndraws = 1000;
VARopt.pctg = 90;
[INF,SUP,MED,BAR] = VARirband(VAR,VARopt);%INF脉冲响应上界

%% 脉冲响应分析
figure
for ii = 1:6 % 冲击序号：供给冲击、
    for jj = 1:6 % 变量序号
        subplot(6, 6, (ii - 1) * 6 + jj)
        plot(1:20, MED(:,jj,ii), "Color",'r', 'LineWidth', 3, 'LineStyle', '-')
        hold on
        plot(1:20, INF(:,jj,ii), "Color",'b', 'LineWidth', 1, 'LineStyle', '--')
        hold on
        plot(1:20, SUP(:,jj,ii), "Color",'b', 'LineWidth', 1, 'LineStyle', '--')
        hold on
        plot(1:20, zeros(20, 1), "Color",'k', 'LineWidth', 1, 'LineStyle', ':')
        if ii ==1
            title('1')
        elseif ii == 2
            title('2')
        else
            title('3')
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
plot(tim, data_demean(:,3),"Color",'bk',"LineWidth",2,"LineStyle","-")

legend('石油供给冲击','总需求冲击','专有型需求冲击','实际油价')

axis tight
set(gca,'FontSize',18,'Color','none')
grid on

%% 预测误差方差分解
[FEVD, VAR] = VARfevd(VAR,VARopt);
