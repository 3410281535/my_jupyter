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

function [] = drawimp(fldraw,vt,interval,datetime,formatOut,formataxis) %׷������ʱ��ļ������ͼ��

global m_ns m_nk m_nl m_asvar;

ns = m_ns;  %��¼��ʱ�������ܹ��м���
nk = m_nk;  %��¼�ܹ��м�������
nl = m_nl;  %��¼�ͺ����

mimpr = xlsread('tvpvar_imp.xlsx'); %��ȡ�������
mimpm = mimpr(:, 3:end); %��ȡ���������ֵ

nimp = size(mimpm, 1) / m_ns; %����ά��+3����ʱ�������м���
mline = [0 .5 0; 0 0 1; 1 0 0; 0 .7 .7]; %����һ��mline���󣬷ֱ��¼����������ɫ
vline = {':', '--', '-', '-.'}; %�ֱ�����ߡ����ߡ�ʵ�֡��㻮����������
nline = size(vt, 2); %��¼�����˼�����ͬ���ͺ���

if fldraw==1 %ֻ���ĸ����ڵĲ�ͬ�ͺ��ڵ�������Ӧͼ��
    
    figure
    for i = 1 : nk %��ÿ����������ѭ��
        for j = 1 : nk %��ÿ����������ѭ����Ӧ�Ƿֱ𿴲�ͬ���������������ĳ��
            id = (i-1)*nk + j; %��¼���ǵڼ���ͼ��
            mimp = reshape(mimpm(:, id), nimp, ns)'; %����Ӧ�ı�����ϵĳ������Ϊnimp��ns�еľ���
            subplot(nk, nk, id); %ȷ������������ͼ��λ��    
            for k = 1 : nline %��ÿһ���ͺ��ڿ�ʼ��ͼ
                plot(datetime,mimp(:, vt(k)+1), char(vline(k)),'Color', mline(k, :),'LineWidth',1) %��ͼ����һ�����������ʾ����Y��������X����������ڶ������������ʾ�����ߵķ��ţ����������������ʾ���ǽ�����������ɫ�����ĸ�������������ߵ���ɫ����
                datetick('x',formataxis);
                hold on
            end
            yl=ylim;
            tstart = datetime(nl+1,1);
            tend = datetime(ns,1)+calmonths(1);
            xlim([tstart tend]);
            if yl(1) * yl(2) < 0 %�������Y��ȡֵ��Χ����X���ͬһ��
                line([tstart, tend], [0, 0], 'Color', ones(1,3)*0.6) %��Y=0���߻�����ע���к�����ķ�Χ
            end
            if id == 1
                timelabel=[interval '֮��'];  
                vlege = timelabel; %�Ե�һ�������ͼ��
                for l = 2 : nline
                    timelabel=[interval '֮��'];
                    vlege = [vlege; timelabel]; %���������������ͼ��
                end
                legend([num2str(vt') vlege])
            end
            hold off
            title(['$\varepsilon_{', char(m_asvar(i)),'}\uparrow\ \rightarrow\ ',char(m_asvar(j)), '$'], 'interpreter', 'latex')
        end
    end
    
elseif fldraw==2 %ֻ���ĸ����ڵĲ�ͬʱ���ʮ�����ڵ�������Ӧͼ��
    
    figure
    for i = 1 : nk %��ÿ����������ѭ��
        for j = 1 : nk %��ÿ����������ѭ����Ӧ�Ƿֱ𿴲�ͬ���������������ĳ��
            id = (i-1)*nk + j; %��¼���ǵڼ���ͼ��
            mimp = reshape(mimpm(:, id), nimp, ns)'; %����Ӧ�ı�����ϵĳ������Ϊnimp��ns�еľ���
            subplot(nk, nk, id); %ȷ������������ͼ��λ��
            for k = 1 : nline
                plot(0:nimp-1, mimp(vt(k), :), char(vline(k)),'Color', mline(k, :),'LineWidth',1) %������0��horizen��Y����ȡ��ʱ�������еĲ�ͬʱ��
                hold on
            end
            vax = axis;
            axis([0 nimp-1 vax(3:4)]) %���ᱣ�ֲ���
            if vax(3) * vax(4) < 0
                line([0, nimp-1], [0, 0], 'Color', ones(1,3)*0.6) %ע���к�����ķ�Χ
            end
            Timelabel=[];
            for l=1:size(vt,2)
                Timelabel=[Timelabel;datetime(vt(1,l),1)];
            end
            if id == 1
                vlege = ['���������'];
                for l = 2 : nline
                    vlege = [vlege; '���������'];
                end
                legend([vlege datestr(Timelabel,formatOut)])
            end
            hold off
            title(['$\varepsilon_{', char(m_asvar(i)),'}\uparrow\ \rightarrow\ ',char(m_asvar(j)), '$'], 'interpreter', 'latex')
        end
    end
    
else %�ֱ�ȫ����������ά������Ӧͼ
    
    for i = 1 : nk %��ÿ����������ѭ��
        for j = 1 : nk %��ÿ����������ѭ����Ӧ�Ƿֱ𿴲�ͬ���������������ĳ��
            figure
            id = (i-1)*nk + j; %��¼���ǵڼ���ͼ��
            mimp = reshape(mimpm(:, id), nimp, ns)'; %����Ӧ�ı�����ϵĳ������Ϊnimp��ns�еľ���
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
