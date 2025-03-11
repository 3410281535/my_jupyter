%% VAR(12) ���������ģ�ͣ�ͨ�͡�GDP��M2
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
% �������
titles = {'GDPƽ��ָ��ʱ������ͼ', 'ʵ��GDP����������ʱ������ͼ', 'M2��������������ʱ������ͼ'};
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
   
%% ��ʽVAR����
nlag = 1; % �ͺ����
const = 1; % �нؾ���

[VAR, VARopt] = VARmodel(data, nlag, const); 

%% ������Լ��ʶ��
VARopt.ident = 'oir';
VARopt.nsteps = 20;

% ������Ӧ����
[IRF, VAR] = VARir(VAR, VARopt);

VARopt.ndraws = 1000; % ��������
VARopt.pctg = 90; % ��������

% ����������Ӧ��������������
[INF, SUP, MED, BAR] = VARirband(VAR, VARopt);

%% ������Ӧ����
figure
for ii = 1:3 % 
    for jj = 1:3 % ������ţ�ͨ�͡�GDP��M2
        subplot(3, 3, (ii - 1) * 3 + jj)
        plot(1:20, MED(:, jj, ii), "Color", 'r', 'LineWidth', 3, 'LineStyle', '-'); % ��λ��
        hold on
        plot(1:20, INF(:, jj, ii), "Color", 'b', 'LineWidth', 1, 'LineStyle', '--'); % �½�
        plot(1:20, SUP(:, jj, ii), "Color", 'b', 'LineWidth', 1, 'LineStyle', '--'); % �Ͻ�
        plot(1:20, zeros(20, 1), "Color", 'k', 'LineWidth', 1, 'LineStyle', ':'); % x��
        hold off
        
        % ���ñ���
        if ii == 1
            title('�������', 'FontSize', 14)
        elseif ii == 2
            title('������', 'FontSize', 14)
        else
            title('�������߳��', 'FontSize', 14)
        end
        
        % ����y���ǩ
        ylabel(yLabels{jj}, 'FontSize', 14)
        
        % ����x���ǩ
        xlabel('ʱ�䲽��', 'FontSize', 14)
        
        % ����ͼ������
        set(gca, 'FontSize', 12, 'Color', 'none')
        grid on
    end
end   

%% Ԥ������ֽ�
[FEVD, VAR] = VARfevd(VAR, VARopt);
figure
for ii = 1:3 % �����ţ�������������������������߳��
    for jj = 1:3 % �������
        subplot(3, 3, (ii - 1) * 3 + jj)
        plot(1:20, FEVD(:, jj, ii) * 1000, "Color", colors{ii}, "LineWidth", 2, "LineStyle", "-"); % ����100����չʾ�ٷֱ�
        hold on
        plot(1:20, zeros(20, 1), "Color", 'k', 'LineWidth', 1, 'LineStyle', ':') % x��
        hold off
        
        % ���ñ���
        if ii == 1
            title('��������Ը�������FEVD', 'FontSize', 14)
        elseif ii == 2
            title('�������Ը�������FEVD', 'FontSize', 14)
        else
            title('�������߳���Ը�������FEVD', 'FontSize', 14)
        end
        
        % ����y���ǩ
        ylabel(yLabels{jj}, 'FontSize', 14)
        
        % ����x���ǩ
        xlabel('ʱ�䲽��', 'FontSize', 14)
        
        % ����y�᷶Χ
        ylim([0 1000]); % �ٷֱ���ʽ
        
        % ����ͼ������
        set(gca, 'FontSize', 12)
        grid on
    end
end

%% ��ʷ�ֽ�
[IRF, VAR] = VARir(VAR, VARopt);
HD = VARhd(VAR);

%% �������
work_HD = HD.shock(:,:,1);

data_demean = data - mean(data);

figure
bar(tim, work_HD, 'stacked')
hold on
plot(tim, data_demean(:,1), "Color", 'k', "LineWidth", 2, "LineStyle", "-")

% ͼ��˵��
legend('�������', '������', '�������߳��', 'GDPƽ��ָ��')

% ����x���ǩ
xlabel('ʱ��', 'FontSize', 18)

% ����y���ǩ
ylabel('��ʷ�ֽ⹱��', 'FontSize', 18)

axis tight
set(gca, 'FontSize', 18, 'Color', 'none')
grid on

%% ������
work_HD = HD.shock(:,:,2);

data_demean = data - mean(data);

figure
bar(tim, work_HD, 'stacked')
hold on
plot(tim, data_demean(:,2), "Color", 'k', "LineWidth", 2, "LineStyle", "-")

% ͼ��˵��
legend('�������', '������', '�������߳��', 'ʵ��GDP')

% ����x���ǩ
xlabel('ʱ��', 'FontSize', 18)

% ����y���ǩ
ylabel('��ʷ�ֽ⹱��', 'FontSize', 18)

axis tight
set(gca, 'FontSize', 18, 'Color', 'none')
grid on

%% �������߳��
work_HD = HD.shock(:,:,3);

data_demean = data - mean(data);

figure
bar(tim, work_HD, 'stacked')
hold on
plot(tim, data_demean(:,3), "Color", 'k', "LineWidth", 2, "LineStyle", "-")

% ͼ��˵��
legend('�������', '������', '�������߳��', 'M2����')

% ����x���ǩ
xlabel('ʱ��', 'FontSize', 18)

% ����y���ǩ
ylabel('��ʷ�ֽ⹱��', 'FontSize', 18)

axis tight
set(gca, 'FontSize', 18, 'Color', 'none')
grid on