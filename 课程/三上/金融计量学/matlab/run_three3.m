clc
clear
addpath('E:\jupyter_program\�γ�\���ڼ���ѧ\matlab\BVAR_-master\BVAR_-master\bvartools')
addpath('E:\jupyter_program\�γ�\���ڼ���ѧ\matlab\BVAR_-master\BVAR_-master\cmintools')

% load the data
load Three.txt

%% HP 滤波
% three  options : lambda=1600, lambda=4, lambda=51200
lam = 1600;   % standard  cycles
HPtrend = Hpfilter(log(Three),lam);
HPcycle = (log(Three)-HPtrend)*100;


%% Sign restri.
lags        = 1;
options.hor = 20;

options.signs{1} = 'y(1,1:2,1)<0'; % ���������ͨ������
options.signs{2} = 'y(2,1:2,1)>0'; % ���������GDP

options.signs{3} = 'y(1,1:2,2)>0'; % ��������ͨ������
options.signs{4} = 'y(2,1:2,2)>0'; % ��������GDP
options.signs{5} = 'y(3,1:2,2)<0'; % ��������M2

options.signs{6} = 'y(1,1:2,3)>0'; % ���߳����ͨ������
options.signs{7} = 'y(2,1:2,3)>0'; % ���߳����GDP
options.signs{8} = 'y(3,1:2,3)>0'; % ���߳����M2


options.K        = 1000;

% run the BVAR
bvar             = bvar_(HPcycle, lags, options);

%% Impulse response
MED = prctile(bvar.irsign_draws, 50, 4);
SUP = prctile(bvar.irsign_draws, 84, 4);
INF = prctile(bvar.irsign_draws, 16, 4);
