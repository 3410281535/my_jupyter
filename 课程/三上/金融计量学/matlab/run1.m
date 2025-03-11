% ��ʼ������
clearvars -except
clc

%% ���빤����
toolboxPaths = {'E:\jupyter_program\�γ�\���ڼ���ѧ\matlab\VAR_Toolbox_2.0\VAR', 
               'E:\jupyter_program\�γ�\���ڼ���ѧ\matlab\VAR_Toolbox_2.0\Utils',
               'E:\jupyter_program\�γ�\���ڼ���ѧ\matlab\VAR_Toolbox_2.0\Stats',
               'E:\jupyter_program\�γ�\���ڼ���ѧ\matlab\VAR_Toolbox_2.0\Auxiliary'};
for i = 1:length(toolboxPaths)
    addpath(toolboxPaths{i});
end

%% ���ݴ���
load('work.txt');
���������� = 100 * diff(log('work')); % ����������ת��

%% ��������
������ǩ = {'GDP Deflator Growth Rate', 'Real GDP Growth Rate', 'M2 Growth Rate'};
ͼ����� = {'GDP Deflator Time Series', 'Real GDP Growth Rate Time Series', 'M2 Time Series'};
��ɫ���� = {'b', 'g', 'r'};
ʱ������ = 1992:0.25:2024;

%% ���ݿ��ӻ�
figure;
for i = 1:3
    subplot(3,1,i);
    plot(time����, ����������(:,i), 'Color', ��ɫ����{i}, 'LineWidth', 2);
    xlabel('Time');
    ylabel(������ǩ{i});
    title(ͼ�����{i});
    grid on;
end

%% VARģ�͹���
�ͺ���� = 1;
�����ؾ� = 1;
[ģ��, ģ��ѡ��] = VARmodel(����������, �ͺ����, �����ؾ�);

%% ����Լ��ʶ��
ģ��ѡ��.ident = 'oir';
ģ��ѡ��.nsteps = 20;

% ������Ӧ����
[��Ӧ����, ģ��] = VARir(ģ��, ģ��ѡ��);

% ��������
[�����½磬�����Ͻ磬��λ����������] = VARirband(ģ��, ģ��ѡ��);

%% ������Ӧ����
figure;
for i = 1:3
    for j = 1:3
        subplot(3, 3, (i - 1) * 3 + j);
        plot(1:20, ��λ��(:, j, i), 'Color', 'r', 'LineWidth', 3);
        hold on;
        plot(1:20, �����½�(:, j, i), 'Color', 'b', 'LineWidth', 1, 'LineStyle', '--');
        plot(1:20, �����Ͻ�(:, j, i), 'Color', 'b', 'LineWidth', 1, 'LineStyle', '--');
        plot(1:20, zeros(20, 1), 'Color', 'k', 'LineWidth', 1, 'LineStyle', ':');
        hold off;
        
        % ���ñ���ͱ�ǩ
        if i == 1
            title('Supply Shock', 'FontSize', 14);
        elseif i == 2
            title('Demand Shock', 'FontSize', 14);
        else
            title('Monetary Policy Shock', 'FontSize', 14);
        end
        ylabel(������ǩ{j}, 'FontSize', 14);
        xlabel('Steps', 'FontSize', 14);
        set(gca, 'FontSize', 12, 'Color', 'none');
        grid on;
    end
end

%% Ԥ������ֽ�
[����ֽ⣬ ģ��] = VARfevd(ģ��, ģ��ѡ��);
figure;
for i = 1:3
    for j = 1:3
        subplot(3, 3, (i - 1) * 3 + j);
        plot(1:20, ����ֽ�(:, j, i) * 1000, 'Color', ��ɫ����{i}, 'LineWidth', 2);
        hold on;
        plot(1:20, zeros(20, 1), 'Color', 'k', 'LineWidth', 1, 'LineStyle', ':');
        hold off;
        
        % ���ñ���ͱ�ǩ
        if i == 1
            title('Supply Shock FEVD', 'FontSize', 14);
        elseif i == 2
            title('Demand Shock FEVD', 'FontSize', 14);
        else
            title('Monetary Policy Shock FEVD', 'FontSize', 14);
        end
        ylabel(������ǩ{j}, 'FontSize', 14);
        xlabel('Steps', 'FontSize', 14);
        ylim([0 1000]);
        set(gca, 'FontSize', 12);
        grid on;
    end
end

%% ��ʷ�ֽ�
[��Ӧ������ ģ��] = VARir(ģ��, ģ��ѡ��);
��ʷ�ֽ� = VARhd(ģ��);

%% ���������ʷ�ֽ�
������� = ��ʷ�ֽ�.shock(:,:,1);
ȥ�������� = ���������� - mean(����������);

figure;
bar(time����, �������, 'stacked');
hold on;
plot(time����, ȥ��������(:,1), 'Color', 'k', 'LineWidth', 2);
legend('Supply Shock', 'Demand Shock', 'Monetary Policy Shock', 'GDP Deflator');
xlabel('Time', 'FontSize', 18);
ylabel('Historical Decomposition', 'FontSize', 18);
axis tight;
set(gca, 'FontSize', 18, 'Color', 'none');
grid on;

%% ��������ʷ�ֽ�
������ = ��ʷ�ֽ�.shock(:,:,2);
ȥ�������� = ���������� - mean(����������);

figure;
bar(time����, ������, 'stacked');
hold on;
plot(time����, ȥ��������(:,2), 'Color', 'k', 'LineWidth', 2);
legend('Supply Shock', 'Demand Shock', 'Monetary Policy Shock', 'Real GDP');
xlabel('Time', 'FontSize', 18);
ylabel('Historical Decomposition', 'FontSize', 18);
axis tight;
set(gca, 'FontSize', 18, 'Color', 'none');
grid on;

%% �������߳����ʷ�ֽ�
�������߳�� = ��ʷ�ֽ�.shock(:,:,3);
ȥ�������� = ���������� - mean(����������);

figure;
bar(time����, �������߳��, 'stacked');
hold on;
plot(time����, ȥ��������(:,3), 'Color', 'k', 'LineWidth', 2);
legend('Supply Shock', 'Demand Shock', 'Monetary Policy Shock', 'M2');
xlabel('Time', 'FontSize', 18);
ylabel('Historical Decomposition', 'FontSize', 18);
axis tight;
set(gca, 'FontSize', 18, 'Color', 'none');
grid on;