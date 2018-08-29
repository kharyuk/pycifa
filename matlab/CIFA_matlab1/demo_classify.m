clear;clc;
load YaleB_32x32;
fea=fea';
RUN=5;

if strcmpi(class(fea),'uint8')||max(fea(:)>100)
    fea=fea./max(fea(:));
end
Ti=sum(gnd==1);
CLS=unique(gnd);
clc;
clear ac ;


train_ratio=0.2:0.1:0.6;

idx=1;
T=size(fea,2);
for r=train_ratio
    for run=1:RUN
        clear teFea teLs trLs;
        fprintf('r=%f  run=%d/%d\n',r,run,RUN);
        
        % Randomly generate training set and test set
        cidx=1;
        for cidx=1:numel(CLS)
            c=CLS(cidx);
            Ti=sum(gnd==c);
            trTi=floor(Ti*r);
            feac=fea(:,gnd==c);
            teTi=sum(gnd==c)-trTi;
            o=randperm(Ti);
            trFea{cidx}=feac(:,o(1:trTi));
            trLs{cidx}=repmat(cidx,1,size(trFea{cidx},2));
            teFea{cidx}=feac(:,o(trTi+1:end));
            teLs{cidx}=repmat(cidx,1,size(teFea{cidx},2));
            cidx=cidx+1;
        end       
        
        %% A -- training
        A=[trFea{:}]';
        teFea=[teFea{:}]';
        teLs=[teLs{:}];
        teLs=reshape(teLs,numel(teLs),1);
        trLs=[trLs{:}];
        trLs=reshape(trLs,numel(trLs),1);
        
        
        
        %% Running classification algorithms
        % teFea: test data
        %     A: training data
        %  trLs: labels of training data
        %% cobe based classifier
        algidx=1;
        cobe_opts.subgroups=max(2,floor(trTi/200));
        cobe_opts.cobeAlg='cobe';
        p=cobe_classify(teFea,A,trLs,cobe_opts);
        ac(idx,run,algidx)=(sum(teLs==p)/numel(teLs))*100;
        
        
        %% KNN
        algidx=2;
        [u d v]=svds(A,50,'L');
        le=knnclassify(teFea*v,A*v,trLs,5,'correlation');
        ac(idx,run,algidx)=(sum(teLs==le)/numel(teLs))*100;
        
        
        
        %% LDA classify diaglinear
        algidx=3;
        [u d v]=svds(A,50,'L');
        le=classify(teFea*v,A*v,trLs,'linear');
        ac(idx,run,algidx)=(sum(teLs==le)/numel(teLs))*100;
    end % run
    idx=idx+1;
end

mac=squeeze(mean(ac,2));
stds=squeeze(std(ac,[],2));



cs=[10 36 106;216 41 0;0 0 255]./255;
figure('Name','Classification of Yale-B');
for idx=1:3
    errorbar(train_ratio,mac(:,idx),stds(:,idx),'Color',cs(idx,:),'LineWidth',2);
    hold on;
end
axis tight;
grid on;
set(gca,'XTick',train_ratio,'XTickLabel',train_ratio);
ylim([30 100]);xlim([0.15 0.65]);
legend({'COBEc','KNN','LDA'},'Location','NorthWest');
xlabel('Size of training data (%)');
ylabel('Accuracy (%)');

