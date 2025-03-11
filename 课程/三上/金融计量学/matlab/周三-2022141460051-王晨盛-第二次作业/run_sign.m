%% VAR(1) ���������ģ�ͣ�ͨ�͡�GDP��M2
clear
clc
tic

%% ���ع�����
addpath('E:\jupyter_program\�γ�\���ڼ���ѧ\matlab\VAR_Toolbox_2.0\VAR')
addpath('E:\jupyter_program\�γ�\���ڼ���ѧ\matlab\VAR_Toolbox_2.0\Utils')
addpath('E:\jupyter_program\�γ�\���ڼ���ѧ\matlab\VAR_Toolbox_2.0\Stats')
addpath('E:\jupyter_program\�γ�\���ڼ���ѧ\matlab\VAR_Toolbox_2.0\Auxiliary')
 
%% ��ȡ����
load work.txt
data = 100 * diff(log(work)); % ת��Ϊ����������

%%
% ����y���ǩ
yLabels = {'GDPƽ��ָ������������', 'ʵ��GDP����������', 'M2��������������'};
ylabels = {'GDPƽ��ָ��', 'ʵ��GDP', 'M2����'};
% �������
titles = {'GDPƽ��ָ��ʱ������ͼ', 'ʵ��GDP����������ʱ������ͼ', 'M2��������������ʱ������ͼ'};
title_shock = {'�������', '������', '�������߳��'}; 
% ��ɫ����
colors = {'b', 'g', 'r'};
tim = 1992:0.25:2024.00;

% ���ݿ��ӻ�
figure 
for ii = 1:3
    subplot(3,1,ii);
    plot(tim, data(:,ii), "Color", colors{ii}, "LineWidth", 2, "LineStyle", "-");
    xlabel('ʱ��');
    ylabel(yLabels{ii});
    title(titles{ii});
    grid on;
end

%% ��ʽVARģ�͹���
nlag  = 1; % �ͺ����
const = 1; % �ؾ�

[VAR, VARopt] = VARmodel(data,nlag,const);

%% ����Լ����ʶ��

% ���裺
% 1.���������ͨ�ͶԹ��������            1-4����ӦΪ��
%              
%             M2�����Թ��������          1-4����ӦΪ��
%
% 2.��������
%             
%
%             
% 3. �������߳����ͨ�ͶԻ������߳����   2-4����ӦΪ��
%                  ʵ��GDP�Ի������߳����1-4����ӦΪ��
%                  M2�����Ի������߳���� 1-4����ӦΪ��



% �������
R(:,:,1) = [1, 4,  1;
            0, 0,  0;
            1, 4,  1 ];
% ������
R(:,:,2) = [0, 0,  0;
            0, 0,  0;
            0, 0,  0 ];
% �������߳��
R(:,:,3) = [2, 4,  1; % ������141
            1, 4,  1;
            1, 4,  1 ];
        
VARopt.nsteps = 20; % ����Ԥ�ⲽ��Ϊ 20 ������
VARopt.ndraws = 1000; % ���ó�������Ϊ 1000 ��

% ���з���Լ��ʶ�𲢼��� IRFs, FEVD, HD
SRout = SR(VAR,R,VARopt);

%% ������Ӧ����
figure
for ii = 1:3 % �������&����
    for jj = 1:3 % % ������&����
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

%% ����ֽ����
figure
for ii = 1:3 % �������&����
    for jj = 1:3 % ֻ��һ��������������߳����
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

%% ��ʷ�ֽ�
GDPDE_HD = squeeze(SRout.HDmed(:,1,:));
REGDP_HD = squeeze(SRout.HDmed(:,2,:));
M2_HD = squeeze(SRout.HDmed(:,3,:));
data_demean = data - mean(data);

% GDPƽ��ָ��
figure 
bar(tim,GDPDE_HD,'stacked')
hold on
plot(tim,data_demean(:,1),"Color",'k','LineWidth',2,'LineStyle','-')

legend('������','�������', '�������߳��','GDPƽ��ָ��')
legend('boxoff')
axis tight
set(gca,'FontSize',18,'Color','none')
grid on

% ʵ��GDP
figure 
bar(tim,REGDP_HD,'stacked')
hold on
plot(tim,data_demean(:,2),"Color",'k','LineWidth',2,'LineStyle','-')

legend('������','�������', '�������߳��','ʵ��GDP')
legend('boxoff')
axis tight
set(gca,'FontSize',18,'Color','none')
grid on

% M2����
figure 
bar(tim,M2_HD,'stacked')
hold on
plot(tim,data_demean(:,3),"Color",'k','LineWidth',2,'LineStyle','-')

legend('������','�������', '�������߳��','M2����')
legend('boxoff')
axis tight
set(gca,'FontSize',18,'Color','none')
grid on
%close all
toc