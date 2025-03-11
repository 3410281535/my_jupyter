%% VAR(24) ʯ�Ͳ��������������ʵ���ͼ�������ģ��
clear
clc
tic
%% ���ع�����
addpath('E:\jupyter_program\�γ�\���ڼ���ѧ\matlab\VAR_Toolbox_2.0\VAR')
addpath('E:\jupyter_program\�γ�\���ڼ���ѧ\matlab\VAR_Toolbox_2.0\Utils')
addpath('E:\jupyter_program\�γ�\���ڼ���ѧ\matlab\VAR_Toolbox_2.0\Stats')
addpath('E:\jupyter_program\�γ�\���ڼ���ѧ\matlab\VAR_Toolbox_2.0\Auxiliary')

%% ��ȡ����
load data_oil.txt

%% ����ͼ
figure
for ii = 1:3
    subplot(3, 1, ii)
    tim = 1973+2/12:1/12:2007+12/12% 1996Q1-2024Q3
    plot(tim, data_oil(:,ii), "Color",'r', 'LineWidth', 3, 'LineStyle', '-')
    xlabel('ʱ��')
    if ii == 1
        title('ʯ�Ͳ���')
    elseif ii == 2
        title('�������')
    else
        title('ʵ���ͼ�')
    end
    axis tight
    set(gca, 'FontSize', 20, 'Color', 'none')
    grid on
end


%% ��ʽVAR����
nlag = 24;%�ͺ����
const = 1;%�нؾ���

[VAR, VARopt] = VARmodel(data_oil,nlag,const); 

%% ������Լ��ʶ��
VARopt.ident = 'oir';
VARopt.nsteps = 20;
[IRF, VAR] = VARir(VAR,VARopt);

VARopt.ndraws = 1000;
VARopt.pctg = 90;
[INF,SUP,MED,BAR] = VARirband(VAR,VARopt);%INF������Ӧ�Ͻ�
%% ������Ӧ����
figure
for ii = 1:3 % �����ţ����������
    for jj = 1:3 % �������
        subplot(3, 3, (ii - 1) * 3 + jj)
        plot(1:20, MED(:,jj,ii), "Color",'r', 'LineWidth', 3, 'LineStyle', '-')
        hold on
        plot(1:20, INF(:,jj,ii), "Color",'b', 'LineWidth', 1, 'LineStyle', '--')
        hold on
        plot(1:20, SUP(:,jj,ii), "Color",'b', 'LineWidth', 1, 'LineStyle', '--')
        hold on
        plot(1:20, zeros(20, 1), "Color",'k', 'LineWidth', 1, 'LineStyle', ':')
        if ii ==1
            title('ʯ�͹������')
        elseif ii == 2
            title('��������')
        else
            title('ר����������')
        end
        
        if jj ==1
            ylabel('ʯ�Ͳ���')
            ylim([-25, 15])
            xlim([1, 20])
        elseif jj == 2
            ylabel('�������')
            ylim([-5, 10])
            xlim([1, 20])
        else
            ylabel('ʵ���ͼ�')
            ylim([-8, 12])
            xlim([1, 20])
        end
        set(gca, 'FontSize', 18, 'Color', 'none')
        grid on
   end
end

%% ��ʷ�ֽ�
[IRF, VAR] = VARir(VAR, VARopt);
HD = VARhd(VAR);
oil_HD = HD.shock(:,:,3);

data_demean = data_oil - mean(data_oil);

figure
bar(tim, oil_HD,'stacked')
hold on
plot(tim, data_demean(:,3),"Color",'k',"LineWidth",2,"LineStyle","-")

legend('ʯ�͹������','��������','ר����������','ʵ���ͼ�')

axis tight
set(gca,'FontSize',18,'Color','none')
grid on

%% Ԥ������ֽ�
[FEVD, VAR] = VARfevd(VAR,VARopt);
