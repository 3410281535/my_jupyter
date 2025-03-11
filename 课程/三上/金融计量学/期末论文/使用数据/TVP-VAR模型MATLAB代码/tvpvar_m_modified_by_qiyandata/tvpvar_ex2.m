%%--------------------------------------------------------%%
%%                     tvpvar_ex2.m                       %%
%%--------------------------------------------------------%%

%%
%%  MCMC estimation for Time-Varying Parameter VAR model
%%  with stochastic volatility
%%
%%  tvpvar_ex*.m illustrates MCMC estimation
%%  using TVP-VAR Package
%%  (Data: "tvpvar_ex.xlsx")
%%

clear all;
close all;

my = xlsread('tvpvar_ex.xlsx');  % load data
Time=datetime(importdata('tvpvar_ex_time.xlsx')); % 载入时间标签并存储变量，请保证其列数为1，行数与时间序列数据的样本数相符，并且置于左上角

asvar = {'p'; 'x'; 'i'};    % variable names
nlag = 1;                   % # of lags

setvar('data', my, asvar, nlag);  % set data

setvar('ranseed', 5);       % set ranseed
setvar('intercept', 1);     % set time-varying intercept
setvar('SigB', 1);          % set non-digaonal for Sig_beta
setvar('impulse', 12);      % set maximum length of impulse
                            % (smaller, calculation faster)

     
mcmc(10000,Time,'yyyy');                % MCMC，输入时间标签变量以调整显示格式，第三个参数用来调整坐标轴时间格式

drawimp(1,[1 6 12], '个季度',Time,'','yyyy');       % draw impulse reponse(1)，画脉冲相应图形，第一个参数为1表示输出等间距脉冲响应图形，第二个参数用于指定冲击发生后要观察的间距，第三个参数用来修改显示的时间间隔的标签，第四个参数输入时间标签变量,此处不适用可以随意输入，第五个参数调整坐标轴的时间显示格式
                            
drawimp(2,[40 70 100], '个季度',Time,'yyyy/qq','yyyy');		% draw impulse response(2)，画脉冲相应图形，第一个参数为2表示输出分时点脉冲响应图形，第二个参数用于指定冲击发生的时点，第三个参数不适用于此种情形可随意输入，第四个参数输入时间标签变量，第五个参数用来指定标签的时间格式，第六个参数调整坐标轴的时间显示格式
                            
drawimp(3,[40 70 100], '个季度',Time,'yyyy/qq','yyyy');     %画三维脉冲响应图形，第一个参数不为1和2即可，后面几个参数不适用于此种情形可随意输入     