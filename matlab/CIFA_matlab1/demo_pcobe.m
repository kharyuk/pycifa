%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;

RUN=50;  %% Monte-Carlo Runs
noiseLevel=20; %% Level of additive noise;
N=10;   %% Number of matrices
c=4;    %% Number of common components. 
K=6;    %% Number of individual components
Jn=50;  %% Number of columns of Yn

load Speech4;
Ac=Speech4';
clear Speech4;

T=size(Ac,1);
pdims=100:100:1000;

sirs=zeros(numel(pdims),RUN);
tims=zeros(1,numel(pdims));

bss_opts.NumOfComp=c;
rI=repmat(K,1,N);
opts=struct('tol',1e-6,'maxiter',1000,'c',c);
for run=1:RUN
    fprintf('P-COBE test: Run [%d/%d] ...\n',run,RUN);
    %% Re-generating observations
    Y=cell(1,N);
    for n=1:N
        Y{n}=Ac;
        Y{n}(:,c+1:c+K)=randn(T,K);
        Y{n}=Y{n}*randn(c+K,Jn);
%         Y{n}=awgn(Y{n},noiseLevel,'measured');
        Y{n}=addGaussianNoise(Y{n},noiseLevel);
    end
    
    %% pCOBE
    for pidx=1:numel(pdims)
        opts.pdim=pdims(pidx);
        fprintf('dim=%d  run=%d/[%d]\n',pdims(pidx),run,RUN);
        ts=tic;
        [eBc ex]=pcobe(Y,opts);
        [se ae]=PMFsobi(eBc(:,1:c),bss_opts);
        tims(pidx)=tims(pidx)+toc(ts);
        sirs(pidx,run)=mean(CalcSIR(Ac,se));
    end
    
    
    
end %% run
tims=tims./RUN;
msirs=mean(sirs,2);
disp(' ==== Results ====');
disp(['Reduced dims:       ', num2str(pdims,'%5d '),'  / Total = ',num2str(T)]);
disp(['Averaged SIRs (dB): ', num2str(msirs','%4.1f  ')]);
disp(['Averaged Time  (s): ', num2str(tims,'%4.1f  ')]);

%% plot;
hf=figure('Name','Performance of PCOBE','units','inch','position',[0 0 3.5 2.5]);
movegui(hf,'center');
set(hf,'units','normalized');
hax1=axes;
plot(hax1,pdims,tims,'r>-','Marker','>','MarkerFaceColor','r','MarkerEdgeColor','r');
xlim([50 1050]);ylim([min(tims) max(tims)]);
ylabel('Averaged running time (s)');
set(hax1,'YAxisLocation','left','YColor','r','XTickLabel',[]);

pos=get(hax1,'position');
hax2=axes('position',pos);

%% virtual plot for legend display
plot(-1,-1,'r>-','Marker','>','MarkerFaceColor','r','MarkerEdgeColor','r');
%

hold on;
plot(hax2,pdims,msirs,'bs-','Marker','s','MarkerFaceColor','b','MarkerEdgeColor','b');
xlim([50 1050]);ylim([10 28]);
set(hax2,'Color','none','YAxisLocation','right','YColor','b');
ylabel('Averaged SIR (dB)');
xlabel('{\itI_p},  ({\itI}=5000, {\itJ_n}=50, {\itc}=4.)');
legend({'Running time','SIR'},'location','southeast','color',[1 1 1]);
grid on;

set([hax1 hax2],'position',[0.16 0.18 0.72 0.8]);
% set(hax2,'Ylim',[20 28.5],'YTick',20:28)
% set(hax1,'Ylim',[0.18 0.455])