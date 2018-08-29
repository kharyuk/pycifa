clear;
clc;
tic;


%% ORL
load ORL_32x32;
nList=[30 40];  %% randomly selected clusters

%% PIE
% load PIE_pose27;
% % nList=[30:10:40 68];
% nList=30;

%% YALE
% % load Yale_32x32;
% % nList=11:15;

if strcmpi(class(fea),'uint8')||max(fea(:))>100
    fea=double(fea);
    fea=fea./max(fea(:));
end

RUN=50;


nCol=min(sum(gnd==1),200);    % number of columns in A{n}
nCol=max(nCol,20);

%% basic information
NC=max(gnd);      % number of classes

Ln=numel(nList);
N=numel(gnd)/nCol; % number of groups. A{1,...,N}

idx=0;
for n=nList
    idx=idx+1;
    for run=1:RUN
        fprintf('n=%d  run=%d ...\n',n,run);
        order=randperm(NC);
        order=order(1:n);
        A=cell(n,1);
        gndn=[];
        
        %% get data
        for i=1:n
            ic=order(i);
            flag=gnd==ic;
            A{i}=fea(flag,:)';
            gndn(end+1:end+sum(flag))=i;
        end
        
        %% randperm
        os=randperm(numel(gndn));
        A=[A{:}];
        A=A(:,os);
        gndn=gndn(os)';
        
        g=floor((numel(gndn)-1)/nCol);
        if numel(gndn)-g*nCol<nCol
            g=g-1;
        end
        gps=[repmat(nCol,1,g) numel(gndn)-g*nCol];
        

        %% generating data
        A=mat2cell(A,size(A,1),gps);
        
        
    %% CIFA+tSNE
        opts.c=2;
        [c Q ]=cobe(A,opts); %% 

        %% ifeature
        ifea=[A{:}]-c*cell2mat(Q);
%         [nc nQ]=cnfe(c,Q);
%         ifea=[A{:}]-nc*cell2mat(nQ);
        ifea=tsne(ifea', [], 2);
        le=kmeans(ifea,n,'replicate',20);
        ac(idx,run,1)=accuracy(gndn,le);
        mu(idx,run,1)=MutualInfo(gndn,le);
        
        
        %% tSNE
        ifea=tsne([A{:}]', [], 2);
        le=kmeans(ifea,n,'replicate',50);
        ac(idx,run,2)=accuracy(gndn,le);
        mu(idx,run,2)=MutualInfo(gndn,le);
        
        %% PCA
%         ifea=compute_mapping([A{:}]','PCA',50);
        [ifea Lmat]=princomp([A{:}],'econ');
        ifea=ifea(:,1:50);
        le=kmeans(ifea,n,'replicate',20);
        ac(idx,run,3)=accuracy(gndn,le);
        mu(idx,run,3)=MutualInfo(gndn,le);

        %% GNMF
         gfea=NormalizeFea([A{:}]');
         gopts=struct('WeightMode','Binary','maxIter',100,'alpha',100);
         W=constructW(gfea,gopts);
         [U V]=GNMF(gfea',n,W,gopts);
         rand('twister',5489);
         glb=kmeans(V,n,'replicate',20);    
         ac(idx,run,4)=accuracy(gndn,glb);
         mu(idx,run,4)=MutualInfo(gndn,glb);
         
         %% MMC
         Qini=rand(size(W,1),n);
         Qmc=mmc_nonnegative(W,Qini);
         [u mle]=max(Qmc');
         ac(idx,run,5)=accuracy(gndn,mle');
         mu(idx,run,5)=MutualInfo(gndn,mle');
    end
end
toc;
% info='RUN 50 from demo_face. nc=2 pca=50. order=cobe-pca-tsne-gnmf-mmc tsne=2';
% save PIE27_120507_a5.mat  ac mu info nList RUN

algstr=['COBE    tSNE      PCA      GNMF      MMC'];
if numel(nList)==1
    mmu=squeeze(mean(mu,2))';
    mac=squeeze(mean(ac,2))';
else
    mmu=squeeze(mean(mu,2));
    mac=squeeze(mean(ac,2));
end
    
clc;
disp('    ==== Averaged Accuracy ====');
disp(algstr);
disp(num2str(mac,'%4.2f    '));
fprintf('\n');
disp('    ==== Averaged MutualInfo ====');
disp(algstr);
disp(num2str(mmu,'%4.2f     '));



