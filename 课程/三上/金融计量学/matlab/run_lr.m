%% VAR(24) GDP��ƽ��ָ��
clear
clc
tic

%% ���ع�����
addpath('E:\jupyter_program\�γ�\���ڼ���ѧ\matlab\VAR_Toolbox_2.0\VAR')
addpath('E:\jupyter_program\�γ�\���ڼ���ѧ\matlab\VAR_Toolbox_2.0\Utils')
addpath('E:\jupyter_program\�γ�\���ڼ���ѧ\matlab\VAR_Toolbox_2.0\Stats')
addpath('E:\jupyter_program\�γ�\���ڼ���ѧ\matlab\VAR_Toolbox_2.0\Auxiliary')

%% ��ȡ����
load GDP_DEF.txt %1992Q1-2024Q2

data = 100 * diff(log(GDP_DEF)); % ת��Ϊ����������

%% ��ʽVARģ�͹���
nlag = 1;%�ͺ����
const = 1;%�ؾ���

[VAR, VARopt] = VARmodel(data,nlag,const); 

%% ������Լ��ʶ��
VARopt.ident = 'bq'; %������Լ��
VARopt.nsteps = 20;
[IRF, VAR] = VARir(VAR,VARopt);

VARopt.ndraws = 1000
VARopt.pctg = 90
[INF,SUP,MED,BAR] = VARirband(VAR,VARopt);%INF������Ӧ�Ͻ�

%% ������Ӧ��������������
VARopt.ndraws = 1000;
VARopt.pctg = 90;
[INF, SUP, MED, BAR] = VARirband(VAR, VARopt); % ����������Ӧ���������䣨�Ͻ�INF���½�SUP����λ��MED��

figure
for ii = 1:size(data, 2) % ��ÿ���������
    for jj = 1:size(data, 2) % ��ÿ����Ӧ����
        subplot(2, 2, (ii-1)*2+jj) % ����������������������2��2�е���ͼ����
        plot(1:VARopt.nsteps, MED(:, jj, ii), "Color", 'r', 'LineWidth', 3, 'LineStyle', '-') % ������Ӧ��λ��
        hold on
        plot(1:VARopt.nsteps, INF(:, jj, ii), "Color", 'b', 'LineWidth', 1, 'LineStyle', '--') % ������Ӧ90%����������Ͻ�
        plot(1:VARopt.nsteps, SUP(:, jj, ii), "Color", 'b', 'LineWidth', 1, 'LineStyle', '--') % ������Ӧ90%����������½�
        plot(1:VARopt.nsteps, zeros(VARopt.nsteps, 1), "Color", 'k', 'LineWidth', 1, 'LineStyle', ':') % x������
        hold off
        if ii == 1
            xlabel('����')
            if jj == 1
                title('GDP�����GDP��Ӱ��')
                ylabel('GDP���������ʱ仯')
            else
                title('GDP�����ƽ��ָ����Ӱ��')
                ylabel('ƽ��ָ�����������ʱ仯')
            end
        elseif ii == 2
            xlabel('����')
            if jj == 1
                title('ƽ��ָ�������GDP��Ӱ��')
                ylabel('GDP���������ʱ仯')
            else
                title('ƽ��ָ�������ƽ��ָ����Ӱ��')
                ylabel('ƽ��ָ�����������ʱ仯')
            end
        end
        set(gca, 'FontSize', 20, 'Color', 'none', 'Box', 'off')
        grid on
    end
end
toc