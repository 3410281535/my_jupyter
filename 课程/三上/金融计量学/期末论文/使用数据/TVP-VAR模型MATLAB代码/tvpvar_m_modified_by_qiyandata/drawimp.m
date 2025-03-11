%%--------------------------------------------------------%%
%%                    TVP-VAR package                     %%
%%--------------------------------------------------------%%
%%
%%  [] = drawimp(vt, fldraw)
%%
%%  "drawimp" draws time-varying impulse response
%%
%%  [input]
%%   (fldraw = 1)
%%     vt:   m*1 vector of horizons to draw impulse
%%   (fldraw = 0)
%%     vt:   m*1 vector of time points to draw impulse
%%

function [] = drawimp(fldraw,vt,interval,datetime,formatOut,formataxis) %追加输入时间的间隔用于图例

global m_ns m_nk m_nl m_asvar;

ns = m_ns;  %记录有时间序列总共有几期
nk = m_nk;  %记录总共有几个变量
nl = m_nl;  %记录滞后阶数

mimpr = xlsread('tvpvar_imp.xlsx'); %读取脉冲矩阵
mimpm = mimpr(:, 3:end); %提取脉冲矩阵数值

nimp = size(mimpm, 1) / m_ns; %数据维度+3除以时间序列有几期
mline = [0 .5 0; 0 0 1; 1 0 0; 0 .7 .7]; %定义一个mline矩阵，分别记录四种线条颜色
vline = {':', '--', '-', '-.'}; %分别定义点线、虚线、实现、点划线四种线形
nline = size(vt, 2); %记录输入了几个不同的滞后期

if fldraw==1 %只画四个以内的不同滞后期的脉冲响应图形
    
    figure
    for i = 1 : nk %对每个变量进行循环
        for j = 1 : nk %对每个变量进行循环，应是分别看不同变量对其他变量的冲击
            id = (i-1)*nk + j; %记录这是第几个图形
            mimp = reshape(mimpm(:, id), nimp, ns)'; %将对应的变量组合的冲击重组为nimp列ns行的矩阵
            subplot(nk, nk, id); %确定它在最后的子图的位置    
            for k = 1 : nline %对每一个滞后期开始做图
                plot(datetime,mimp(:, vt(k)+1), char(vline(k)),'Color', mline(k, :),'LineWidth',1) %作图，第一个输入变量表示的是Y变量（无X轴变量），第二个输入变量表示的是线的符号，第三个输入变量表示的是接下来定义颜色，第四个输入变量就是线的颜色参数
                datetick('x',formataxis);
                hold on
            end
            yl=ylim;
            tstart = datetime(nl+1,1);
            tend = datetime(ns,1)+calmonths(1);
            xlim([tstart tend]);
            if yl(1) * yl(2) < 0 %如果出现Y的取值范围不在X轴的同一侧
                line([tstart, tend], [0, 0], 'Color', ones(1,3)*0.6) %将Y=0的线画出，注意有横坐标的范围
            end
            if id == 1
                timelabel=[interval '之后'];  
                vlege = timelabel; %对第一条线添加图例
                for l = 2 : nline
                    timelabel=[interval '之后'];
                    vlege = [vlege; timelabel]; %对其他几条线添加图例
                end
                legend([num2str(vt') vlege])
            end
            hold off
            title(['$\varepsilon_{', char(m_asvar(i)),'}\uparrow\ \rightarrow\ ',char(m_asvar(j)), '$'], 'interpreter', 'latex')
        end
    end
    
elseif fldraw==2 %只画四个以内的不同时点的十二期内的脉冲响应图形
    
    figure
    for i = 1 : nk %对每个变量进行循环
        for j = 1 : nk %对每个变量进行循环，应是分别看不同变量对其他变量的冲击
            id = (i-1)*nk + j; %记录这是第几个图形
            mimp = reshape(mimpm(:, id), nimp, ns)'; %将对应的变量组合的冲击重组为nimp列ns行的矩阵
            subplot(nk, nk, id); %确定它在最后的子图的位置
            for k = 1 : nline
                plot(0:nimp-1, mimp(vt(k), :), char(vline(k)),'Color', mline(k, :),'LineWidth',1) %横轴是0到horizen，Y就是取的时间序列中的不同时点
                hold on
            end
            vax = axis;
            axis([0 nimp-1 vax(3:4)]) %横轴保持不变
            if vax(3) * vax(4) < 0
                line([0, nimp-1], [0, 0], 'Color', ones(1,3)*0.6) %注意有横坐标的范围
            end
            Timelabel=[];
            for l=1:size(vt,2)
                Timelabel=[Timelabel;datetime(vt(1,l),1)];
            end
            if id == 1
                vlege = ['冲击发生在'];
                for l = 2 : nline
                    vlege = [vlege; '冲击发生在'];
                end
                legend([vlege datestr(Timelabel,formatOut)])
            end
            hold off
            title(['$\varepsilon_{', char(m_asvar(i)),'}\uparrow\ \rightarrow\ ',char(m_asvar(j)), '$'], 'interpreter', 'latex')
        end
    end
    
else %分别画全部变量的三维脉冲响应图
    
    for i = 1 : nk %对每个变量进行循环
        for j = 1 : nk %对每个变量进行循环，应是分别看不同变量对其他变量的冲击
            figure
            id = (i-1)*nk + j; %记录这是第几个图形
            mimp = reshape(mimpm(:, id), nimp, ns)'; %将对应的变量组合的冲击重组为nimp列ns行的矩阵
            x=datetime(nl+1:end,1);
            y=0:nimp-1;
            Z=mimp(nl+1:end,:)';
            [X,Y] = meshgrid(x,y);
            mesh(Y,X,Z)
            datetick('y', formataxis);
            set(gca,'YDir','reverse');
            set(gca,'XDir','reverse');
            title(['$\varepsilon_{', char(m_asvar(i)),'}\uparrow\ \rightarrow\ ',char(m_asvar(j)), '$'], 'interpreter', 'latex')
        end
    end
end
