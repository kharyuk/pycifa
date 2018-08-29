clear;
clc;
commandwindow;

RUN=2;  %% Monte-Carlo Runs
c=4;    %% Number of common components
N=10;   %% Number of matrices
K=6;    %% Number of individual components
Jn=50;  %% Number of columns of Yn
noiseLevel=20; %% Level of additive noise;

load Speech4;
Ac=Speech4;
clear Speech4;

% you may change the variance of Ac, which often leads to different
%   results.
reScaleSources=500;  %% LARGER one leads to better performance. 
Ac=bsxfun(@rdivide,Ac',sum(Ac.^2,2)'.^.5)';
Ac=Ac.*sqrt(reScaleSources);


ALG=5;

T=size(Ac,2);

sirs=zeros(RUN,c,ALG);
tims=zeros(1,ALG);


bss_opts.NumOfComp=c;
rI=repmat(K,1,N);
cobe_opts=struct('epsilon',.03,'maxiter',500,'c',c,'PCAdim',repmat(K+c,1,N));
cobec_opts=struct('c',c,'maxiter',500);

for run=1:RUN
    fprintf('Run [%d/%d] ...\n',run,RUN);
    
    %% Re-generating observations
    Y0=cell(1,N);
    Y=cell(1,N);
    for n=1:N
        Y{n}(1:c,1:T)=Ac;
        Y{n}(c+1:c+K,:)=randn(K,T);             
        
        Y{n}=randn(Jn,c+K)*Y{n};
        Y0{n}=Y{n}';  %% for c-detection
%         Y{n}=awgn(Y{n},noiseLevel,'measured');
        Y{n}=addGaussianNoise(Y{n},noiseLevel);
    end


    disp('JIVE is running ...');
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    % %[r,rIndiv] = JIVE_RankSelect(Y,.05,100);
    algindex=1;
    ts=tic;
    [J,X] = JIVE(Y,c,rI,'y',1e-5,1000);
    [w, h]=PMFsobi(J',bss_opts);
    tims(algindex)=tims(algindex)+toc(ts);
    sirs(run,:,algindex)=sort(CalcSIR(Ac',w));
    

    %transpose Y
    Y=cellfun(@(x) x',Y,'uni',false);


    disp('COBE is running ...');    
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    %% COBE
    algindex=2;
    ts=tic;
    [eBc]=cobe(Y,cobe_opts);
    se=PMFsobi(eBc(:,1:c),bss_opts);
    tims(algindex)=tims(algindex)+toc(ts);
    sirs(run,:,algindex)=sort(CalcSIR(Ac',se));
    
    disp('COBEc is running ...');   
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    %% COBEc
    cobe_opts.c=c;
    algindex=3;
    ts=tic;
    [eBc ]=cobec(Y,cobec_opts);
    se=PMFsobi(eBc(:,1:c),bss_opts);
    tims(algindex)=tims(algindex)+toc(ts);
    sirs(run,:,algindex)=sort(CalcSIR(Ac',se));    
    
    
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    %% MCCA   ---- joint BSS
    disp('JBSS is running ...');    
    algindex=4;
    ts=tic;
    Se=call_mcca(Y,c);
    tims(algindex)=tims(algindex)+toc(ts);
    si=zeros(1,c);
    for n=1:N
        si=si+CalcSIR(Ac',real(Se{n}));
    end
    disp(['JBSS - Without SOBI: ' num2str(sort(si./N),'%4.2f    ')]);
    
    si=zeros(1,c);
    for n=1:N
        si=si+CalcSIR(Ac',PMFsobi(Se{n}));
    end
    sirs(run,:,algindex)=sort(si./N);
    disp(['JBSS - With SOBI:    ' num2str(sort(si./N),'%4.2f    ')]);
    
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% PCA 
    algindex=5;
    ts=tic;
    [coe, pcs]=princomp([Y{:}],'econ');
    pcs=pcs(:,1:c);
%     [pcs d v]=svds([Y{:}],c,'L');
    se=PMFsobi(pcs,bss_opts);
    tims(algindex)=tims(algindex)+toc(ts);
    sirs(run,:,algindex)=sort(CalcSIR(Ac',se));    
    
    
end %% run

tims=tims./RUN;
msir=squeeze(mean(sirs,1));
fprintf('\n');
disp(' ============= RESULTS ============');
disp(' --  Mean SIRs (dB) -- ');
disp('JIVE    COBE    COBEc   JBSS    PCA');
disp(num2str(msir,' %4.1f   '));
disp('---------------------');
disp(num2str(mean(msir),' %4.1f   '));
fprintf(' \n \n');
disp('                        JIVE    COBE     COBEc   JBSS    PCA');
disp(['  Time consumption (s):  ' num2str(tims,'%4.1f    ')]);
% save demo_ica.mat sirs tims


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EXP II
%% detection of number of components
%% single test
cobe_opts.c=20;cobe_opts.PCAdim=[];
[estAc, Q, res]=cobe(Y,cobe_opts);
hfig=figure('units','inch','position',[1 1 4.5 3],'visible','off');
idx=1:4;
c1=[10 36 106]./255;
c2=[216 41 0]./255;
plot(idx,res(idx),'.-','Color',c1);
hold on;
idx=5:cobe_opts.c;
plot(idx,res(idx),'x-','Color',c2);
legend({'Common','Individual'},'Location','Southeast');
grid on;
xlabel('Index \iti');
ylabel('$\frac{1}{N}f_i$','interp','latex');
movegui(hfig,'center');
set(get(gca,'child'),'linewidth',1,'MarkerSize',8);
set(hfig,'visible','on');

% % %% Full test
% % cobe_opts.c=20;
% % res=[];
% % noiseLevel=20:-10:0;
% % linestyle={'-','-.',':'};
% % for i=1:numel(noiseLevel)    
% %     fprintf('NoiseLevel: %d dB.\n',noiseLevel(i));
% %     for n=1:N
% %         Y{n}=awgn(Y0{n},noiseLevel(i),'measured');
% %         Y{n}=addGaussianNoise(Y0{n},noiseLevel(i));
% %     end
% %     
% %     [estAc Q res(i,:)]=cobe(Y,cobe_opts);
% % end
% % 
% % 
% % figure('Name','fi');
% % c1=[10 36 106]./255;
% % c2=[216 41 0]./255;
% % drawfunc=@plot;
% % hax1=axes;
% % pos=get(hax1,'position');
% % for i=1:numel(noiseLevel)
% %     idx=1:4;
% %     drawfunc(hax1,idx,res(i,idx),horzcat('s',linestyle{i}),'Color',c1,'MarkerFaceColor','none','MarkerEdgeColor',c1);
% %     hold on;
% % end
% % xlim([1 cobe_opts.c]);
% % ylim([0 .9]);
% % hlgd1=legend({'20dB','10dB','0dB'});
% % 
% % hax2=axes;
% % for i=1:numel(noiseLevel)
% %     idx=5:cobe_opts.c;
% %     drawfunc(hax2,idx,res(i,idx),horzcat('x',linestyle{i}),'Color',c2,'MarkerFaceColor',c2,'MarkerEdgeColor',c2);
% %     hold on;
% % end
% % xlim([1 cobe_opts.c]);
% % ylim([0 .9]);
% % grid on;
% % hlgd2=legend({'20dB','10dB','0dB'},'Location','southeast');
% % set(hax2,'Color','none','position',pos);
% % xlabel('Index \iti');
% % ylabel('$\frac{1}{N}f_i$','interp','latex');
% % pl1=get(hlgd1,'position');
% % pl2=get(hlgd2,'position');
% % pl1=[pl2(1)-pl1(3)-0.005 pl2(2:4)];
% % set(hlgd1,'position',pl1);