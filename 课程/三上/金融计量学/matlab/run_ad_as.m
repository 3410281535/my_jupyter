% VAR(1) ����������������ģ��
clear
clc

tic
%% ���ع���
addpath('E:\jupyter_program\�γ�\���ڼ���ѧ\matlab\VAR_Toolbox_2.0\VAR')
addpath('E:\jupyter_program\�γ�\���ڼ���ѧ\matlab\VAR_Toolbox_2.0\Utils')
addpath('E:\jupyter_program\�γ�\���ڼ���ѧ\matlab\VAR_Toolbox_2.0\Stats')
addpath('E:\jupyter_program\�γ�\���ڼ���ѧ\matlab\VAR_Toolbox_2.0\Auxiliary')
%% ��������
load Two1.txt

%% ��ȡ����
% GDP
figure
subplot(1,2,1)
tim = 1997:0.25:2024.50; % 1997Q1-2024Q3
plot(tim,Two1(:,1),"Color",'r','LineWidth',3,'LineStyle','-')
xlabel('ʱ��')
ylabel('%')
title('GDP')
axis tight
set(gca,'FontSize',20,'Color','none')
grid on

%����
% figure
subplot(1,2,2)
tim = 1997:0.25:2024.50; % 1997Q1-2024Q3
plot(tim,Two1(:,2),"Color",'r','LineWidth',3,'LineStyle','-')
xlabel('ʱ��')
ylabel('%')
title('CPI')
axis tight
set(gca,'FontSize',20,'Color','none')
grid on

%% ��ʽVARģ�͹���
nlag  = 1; % �ͺ����
const = 1; % �ؾ�

[VAR, VARopt] = VARmodel(Two1,nlag,const);

%%  ����Լ����ʶ��

%������
%              from        to          sign
  R(:,:,1) = [ 1           4           1          % GDP
               1           4           1 ];       % CPI

%�������
%              from        to          sign
  R(:,:,2) = [ 1           4           1          % GDP
               1           4           -1 ];       % CPI

VARopt.nsteps = 20;
VARopt.ndraws = 1000;  

SRout = SR(VAR,R,VARopt);

%% ������Ӧ����
figure
for ii = 1:2 % �������&����
    for jj = 1:2 % ������&����
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
            ylabel('������')          
        else   
            ylabel('�������') 
        end        

        axis tight
        set(gca,'FontSize',18,'Color','none')
        grid on

    end
end

%% ����ֽ����
figure
for ii = 1:2 % �������&����
    for jj = 1:2 % ������&����
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
            ylabel('������')  
            ylim([0 1])
            xlim([1 20])
        else   
            ylabel('�������')
            ylim([0 1])
            xlim([1 20])
        end        

        %axis tight
        set(gca,'FontSize',18,'Color','none')
        grid on

    end
end


%% ��ʷ�ֽ�
GDP_HD = squeeze(SRout.HDmed(:,1,:));
CPI_HD = squeeze(SRout.HDmed(:,2,:));

Two1_demean = Two1 - mean(Two1);

figure 
bar(tim,GDP_HD,'stacked')
hold on
plot(tim,Two1_demean(:,1),"Color",'k','LineWidth',2,'LineStyle','-')

legend('������','�������','GDP')
legend('boxoff')
axis tight
set(gca,'FontSize',18,'Color','none')
grid on

%% CPI
figure 
bar(tim,CPI_HD,'stacked')
hold on
plot(tim,Two1_demean(:,2),"Color",'k','LineWidth',2,'LineStyle','-')

legend('������','�������','CPI')
legend('boxoff')
axis tight
set(gca,'FontSize',18,'Color','none')
grid on

