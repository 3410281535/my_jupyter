%%--------------------------------------------------------%%
%%                     tvpvar_ex1.m                       %%
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
Time=datetime(importdata('tvpvar_ex_time.xlsx')); % ����ʱ���ǩ���洢�������뱣֤������Ϊ1��������ʱ���������ݵ�����������������������Ͻ�

asvar = {'p'; 'x'; 'i'};    % variable names
nlag = 2;                   % # of lags

setvar('data', my, asvar, nlag); % set data

setvar('fastimp', 1);       % fast computing of response

mcmc(10000,Time,'yyyy');                % MCMC������ʱ���ǩ�����Ե�����ʾ��ʽ��������������������������ʱ���ʽ

drawimp(1,[4 8 12], '������',Time,'','yyyy');       % draw impulse reponse(1)����������Ӧͼ�Σ���һ������Ϊ1��ʾ����ȼ��������Ӧͼ�Σ��ڶ�����������ָ�����������Ҫ�۲�ļ�࣬���������������޸���ʾ��ʱ�����ı�ǩ�����ĸ���������ʱ���ǩ����,�˴������ÿ����������룬��������������������ʱ����ʾ��ʽ
                            
drawimp(2,[30 60 90], '������',Time,'yyyy/qq','yyyy');		% draw impulse response(2)����������Ӧͼ�Σ���һ������Ϊ2��ʾ�����ʱ��������Ӧͼ�Σ��ڶ�����������ָ�����������ʱ�㣬�����������������ڴ������ο��������룬���ĸ���������ʱ���ǩ�������������������ָ����ǩ��ʱ���ʽ�����������������������ʱ����ʾ��ʽ
                            
drawimp(3,[30 60 90], '������',Time,'yyyy/qq','yyyy');     %����ά������Ӧͼ�Σ���һ��������Ϊ1��2���ɣ����漸�������������ڴ������ο���������                       