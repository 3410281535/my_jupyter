% VAR(1) 产出和利率两变量模型
clear
clc

tic
%% 加载工具箱
addpath('C:\Users\zhuzixiang\Desktop\2024年下半年教学\1_金融计量\Excersie\VAR_Toolbox_2.0\VAR_Toolbox_2.0\VAR')
addpath('C:\Users\zhuzixiang\Desktop\2024年下半年教学\1_金融计量\Excersie\VAR_Toolbox_2.0\VAR_Toolbox_2.0\Utils')
addpath('C:\Users\zhuzixiang\Desktop\2024年下半年教学\1_金融计量\Excersie\VAR_Toolbox_2.0\VAR_Toolbox_2.0\Stats')
addpath('C:\Users\zhuzixiang\Desktop\2024年下半年教学\1_金融计量\Excersie\VAR_Toolbox_2.0\VAR_Toolbox_2.0\Auxiliary')

%% 读取数据
load Two.txt

%% 数据图
% GDP
figure
subplot(1,2,1)
tim = 1997:0.25:2024.50; % 1997Q1-2024Q3
plot(tim,Two(:,1),"Color",'r','LineWidth',3,'LineStyle','-')
xlabel('时间')
ylabel('%')
title('GDP')
axis tight
set(gca,'FontSize',20,'Color','none')
grid on

% 利率
% figure
subplot(1,2,2)
tim = 1997:0.25:2024.50; % 1997Q1-2024Q3
plot(tim,Two(:,2),"Color",'r','LineWidth',3,'LineStyle','-')
xlabel('时间')
ylabel('%')
title('CPI')
axis tight
set(gca,'FontSize',20,'Color','none')
grid on

%% 简化式VAR模型估计
nlag  = 1; % 滞后阶数
const = 1; % 截距项

[VAR, VARopt] = VARmodel(Two,nlag,const);

%% 符号约束的识别

% 需求冲击
%              from        to          sign
  R(:,:,1) = [ 1           4           1          % GDP
               1           4           1 ];       % CPI

% 供给冲击
%              from        to          sign
  R(:,:,2) = [ 1           4           1          % GDP
               1           4          -1 ];       % CPI

VARopt.nsteps = 20;
VARopt.ndraws = 1000;  

SRout = SR(VAR,R,VARopt);

%% 脉冲响应分析
figure
for ii = 1:2 % 变量序号&行序号
    for jj = 1:2 % 冲击序号&列序号
        subplot(2,2,2*(ii-1)+jj)
        plot(1:20,SRout.IRFmed(:,ii,jj),"Color",'r','LineWidth',3,'LineStyle','-')
        hold on
        plot(1:20,SRout.IRFinf(:,ii,jj),"Color",'r','LineWidth',1,'LineStyle','--')
        hold on
        plot(1:20,SRout.IRFsup(:,ii,jj),"Color",'r','LineWidth',1,'LineStyle','--')
        hold on
        plot(1:20,zeros(20,1),"Color",'k','LineWidth',1,'LineStyle',':')

        if ii == 1  
            title('GDP')
        else   
            title('CPI')
        end

        if jj == 1  
            ylabel('需求冲击')          
        else   
            ylabel('供给冲击') 
        end        

        axis tight
        set(gca,'FontSize',18,'Color','none')
        grid on

    end
end

%% 方差分解分析
figure
for ii = 1:2 % 变量序号&行序号
    for jj = 1:2 % 冲击序号&列序号
        subplot(2,2,2*(ii-1)+jj)
        plot(1:20,SRout.FEVDmed(:,ii,jj),"Color",'r','LineWidth',3,'LineStyle','-')
        hold on
        plot(1:20,SRout.FEVDinf(:,ii,jj),"Color",'r','LineWidth',1,'LineStyle','--')
        hold on
        plot(1:20,SRout.FEVDsup(:,ii,jj),"Color",'r','LineWidth',1,'LineStyle','--')
        hold on
        plot(1:20,zeros(20,1),"Color",'k','LineWidth',1,'LineStyle',':')

        if ii == 1  
            title('GDP')
        else   
            title('CPI')
        end

        if jj == 1  
            ylabel('需求冲击')  
            ylim([0 1])
            xlim([1 20])
        else   
            ylabel('供给冲击')
            ylim([0 1])
            xlim([1 20])
        end        

        %axis tight
        set(gca,'FontSize',18,'Color','none')
        grid on

    end
end


%% 历史分解
GDP_HD = squeeze(SRout.HDmed(:,1,:));
CPI_HD = squeeze(SRout.HDmed(:,2,:));

Two_demean = Two - mean(Two);

figure 
bar(tim,GDP_HD,'stacked')
hold on
plot(tim,Two_demean(:,1),"Color",'k','LineWidth',2,'LineStyle','-')

legend('需求冲击','供给冲击','GDP')
legend('boxoff')
axis tight
set(gca,'FontSize',18,'Color','none')
grid on

%% CPI
figure 
bar(tim,CPI_HD,'stacked')
hold on
plot(tim,Two_demean(:,2),"Color",'k','LineWidth',2,'LineStyle','-')

legend('需求冲击','供给冲击','CPI')
legend('boxoff')
axis tight
set(gca,'FontSize',18,'Color','none')
grid on

