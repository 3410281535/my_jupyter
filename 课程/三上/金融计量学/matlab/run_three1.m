% example_3_irf.m Impulse  responses, variance  decomp, historical decomp
%                 plus  time varying  IRFs
% Authors:   Filippo Ferroni and  Fabio Canova
% Date:     27/02/2020, revised  14/12/2020
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computation of IRFs using various  identification schemes
% Calculation of  variance and  historical decompositions
% Rolling window estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all; clc;
clear
addpath('E:\jupyter_program\课程\金融计量学\matlab\BVAR_-master\BVAR_-master\bvartools')
addpath('E:\jupyter_program\课程\金融计量学\matlab\BVAR_-master\BVAR_-master\cmintools')


% load the data
load Three.txt

y = 100*diff(log(Three));
  
%% 2) Sign restrictions
% run the BVAR
lags        = 1;
options.hor = 20;
% Demand shock
options.signs{1} = 'y(1,1:1,1)>0'; % 
options.signs{2} = 'y(2,1:1,1)>0'; % 
options.signs{3} = 'y(3,1:1,1)<0'; % 
% Supply shock
options.signs{4} = 'y(1,1:1,2)<0'; % 
options.signs{5} = 'y(2,1:1,2)>0'; % 
% MP shock
options.signs{6} = 'y(1,1:1,3)>0'; % 
options.signs{7} = 'y(2,1:1,3)>0'; % 
options.signs{8} = 'y(3,1:1,3)>0'; % 

options.K        = 1000;
 
% run the BVAR
bvar2             = bvar_(y,lags,options);

MED      = prctile(bvar2.irsign_draws,50,4);
SUP      = prctile(bvar2.irsign_draws,16,4);
INF      = prctile(bvar2.irsign_draws,84,4);

%% 脉冲响应分析
figure
for ii = 1:3 % 冲击序号
    for jj = 1:3 % 变量序号
        subplot(3,3,3*(ii-1)+jj)

        plot(1:20,MED(jj,:,ii),"Color",'r','LineWidth',3,'LineStyle','-')
        hold on
        plot(1:20,INF(jj,:,ii),"Color",'r','LineWidth',1,'LineStyle','--')
        hold on
        plot(1:20,SUP(jj,:,ii),"Color",'r','LineWidth',1,'LineStyle','--')
        hold on
        plot(1:20,zeros(20,1),"Color",'k','LineWidth',1,'LineStyle',':')

        if ii == 1  
            title('需求冲击')
        elseif ii == 2    
            title('供给冲击')
        else   
            title('货币政策冲击')
        end

        if jj == 1  
            ylabel('通货膨胀')

        elseif jj == 2    
            ylabel('产出')
            
        else   
            ylabel('M2')

        end

        axis tight
        set(gca,'FontSize',18,'Color','none')
        grid on
    end
end

