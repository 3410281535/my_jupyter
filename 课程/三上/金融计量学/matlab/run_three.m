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

HPcycle1 = HPcycle(1:72,:);
HPcycle2 = HPcycle(72:end,:);

%% Sign restri.
lags        = 1;
options.hor = 20;

options.signs{1} = 'y(1,1:2,1)<0'; % 供给冲击对�?�胀
options.signs{2} = 'y(2,1:2,1)>0'; % 供给冲击对GDP

options.signs{3} = 'y(1,1:2,2)>0'; % �?求冲击对通胀
options.signs{4} = 'y(2,1:2,2)>0'; % �?求冲击对GDP
options.signs{5} = 'y(3,1:2,2)<0'; % �?求冲击对M2

options.signs{6} = 'y(1,1:2,3)>0'; % 政策冲击对�?�胀
options.signs{7} = 'y(2,1:2,3)>0'; % 政策冲击对GDP
options.signs{8} = 'y(3,1:2,3)>0'; % 政策冲击对M2

options.K        = 1000;

% run the BVAR
bvar             = bvar_(HPcycle, lags, options);

%% Impulse response
MED = prctile(bvar.irsign_draws, 50, 4);
SUP = prctile(bvar.irsign_draws, 84, 4);
INF = prctile(bvar.irsign_draws, 16, 4);

%% HD

opts_.Omega =  mean(bvar.Omegas,3);
[yDecomp,v] = histdecomp(bvar,opts_);

yDecomp = yDecomp(:,:,1:3);
DEFHD = squeeze(yDecomp(:,1,:));
GDPHD = squeeze(yDecomp(:,2,:));
M2HD  = squeeze(yDecomp(:,3,:));
tim = 1992.25:.25:2024.25;
figure
for ii =1:3

subplot(2,2,ii)
bar(tim,squeeze(yDecomp(:,ii,:)),'stacked')
hold on
plot(tim,HPcycle(2:end,ii),"Color",'k','LineWidth',2,'LineStyle','-')

if ii ==1
    legend('供给冲击','�?求冲�?','政策冲击','通胀')
elseif ii ==2
    legend('供给冲击','�?求冲�?','政策冲击','GDP')
else
    legend('供给冲击','�?求冲�?','政策冲击','M2')
end

legend('boxoff')
axis tight
set(gca,'FontSize',24,'Color','none')
grid on

end


