%% VAR(1)
clear
clc
tic
%% ���ع�����
addpath('E:\jupyter_program\�γ�\���ڼ���ѧ\matlab\VAR_Toolbox_2.0\VAR')
addpath('E:\jupyter_program\�γ�\���ڼ���ѧ\matlab\VAR_Toolbox_2.0\Utils')
addpath('E:\jupyter_program\�γ�\���ڼ���ѧ\matlab\VAR_Toolbox_2.0\Stats')
addpath('E:\jupyter_program\�γ�\���ڼ���ѧ\matlab\VAR_Toolbox_2.0\Auxiliary')

%% ��ȡ����
load Two.txt

%% ����ͼ
figure
tim = 1996:0.25:2024.50;%1996Q1-2024Q3
subplot(1, 2, 1)
plot(tim, Two(:,1), "Color",'r', 'LineWidth', 3, 'LineStyle', '-')
xlabel('ʱ��')
ylabel('%')
title('�й�GDPͬ������')
axis tight
set(gca, 'FontSize', 20, 'Color', 'none')
grid on

subplot(1, 2, 2)
plot(tim, Two(:,2), "Color",'r', 'LineWidth', 3, 'LineStyle', '-')
xlabel('ʱ��')
ylabel('%')
title('ͬҵ�������')
axis tight
set(gca, 'FontSize', 20, 'Color', 'none')
grid on

%% ��ʽVAR����
nlag = 1;%�ͺ����
const = 1;%�нؾ���

[VAR, VARopt] = VARmodel(Two,nlag,const); 

%������Լ��ʶ��
VARopt.ident = 'oir';
VARopt.nsteps = 40;
[IRF, VAR] = VARir(VAR,VARopt);

%%������Ӧ����
%%�������߳����GDP��Ӱ��
figure
plot(1:40,IRF(:,1,2), 'Color','b','LineWidth',3,'LineStyle', '-')
xlabel('��Ӧ����')
ylabel('%')
axis tight
set(gca, 'FontSize', 20, 'Color', 'none')
grid


%%������ 1996Q1-2009Q4, 2010Q1-2024Q3
nlag = 1;
const = 1;

Two = Two(1:end,:);

data1 = Two(1:56,:);%1996-2009
data2 = Two(57:end,:);%2010-2024
data3 = Two;%2010-2024

[VAR1, VARopt1] = VARmodel(data1, nlag, const);
[VAR2, VARopt2] = VARmodel(data2, nlag, const);
[VAR3, VARopt3] = VARmodel(data3, nlag, const);
%% ������Լ��ʶ��
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
xlabel('��Ӧ����')
ylabel('%')
title('�������߳����GDP��Ӱ��')
axis tight
set(gca, 'FontSize', 20, 'Color', 'none')
grid on

%%�ƶ����ڷ���
GDP_to_MP = zeros(40,60);
 for ii = 1:60
    data = Two(1:ii + 55,:); 
    [VAR, VARopt] = VARmodel(data, nlag, const);
    
    [IRF, VAR] = VARir(VAR, VARopt);
    GDP_to_MP_temp = IRF(:,1,2);
    GDP_to_MP(:,ii) = GDP_to_MP_temp;
    ii
 end
 
%% ��ά������Ӧ
[X,Y] = meshgrid(2009.75:0.25:2024.50,1:40);
figure

surf(X,Y,GDP_to_MP)
%axis tight
set(gca, 'FontSize', 20, 'Color', 'none')
grid on

clear
clc
%% ��ʽVAR����
nlag = 1;%�ͺ����
const = 1;%�нؾ���

load Two.txt
Two = Two(1:end - 19,:);

[VAR,VARopt] = VARmodel(Two,nlag,const);

%%�����������
VARopt.ndraws = 1000;%���ó�������
VARopt.pctg = 68;%������������,68Ϊ��С���Ŵ�
VARopt.indent = 'oir';
VARopt.nsteps = 40;

[INF,SUP,MED,BAR] = VARirband(VAR,VARopt);%INF������Ӧ�Ͻ�
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
xlabel('��Ӧ����')
ylabel('%')
title('�������߳����GDP��Ӱ��')
axis tight
set(gca, 'FontSize', 20, 'Color', 'none')
grid on
%VARopt.pctg = 90;ʱ������ 

%%����ֽ�
[FEVD, VAR] = VARfevd(VAR,VARopt);

%%����ֽ����
figure
plot(1:40,FEVD(:,1,2),'Color','k','LineWidth',2,'LineStyle','-')

xlabel('��Ӧ����')
xlim([1,40])
ylim([0, 1])
title('�������߳����GDP�Ľ��ͱ���')

set(gca, 'FontSize', 20, 'Color', 'none')
grid on

%%����ֽ����
figure
plot(1:40,FEVD(:,1,1),'Color','k','LineWidth',2,'LineStyle','-')

xlabel('��Ӧ����')
xlim([1,40])
ylim([0, 1])
title('�������߳�������ʵĽ��ͱ���')
%axis tight
set(gca, 'FontSize', 20, 'Color', 'none')
grid on

toc;
 