function [ labels ] = cobe_classify( testData,training,group,opts)
%COBE_CLASSIFY Summary of this function goes here
%   Detailed explanation goes here

%dist: correlation|Euclid
%cobe: cobe|cobec|pcobe
defopts=struct('c',[],'subgroups',2,'nn',false,'dist','correlation','cobeAlg','pcobe','pdim',[]);
if ~exist('opts','var')
    opts=struct();
end

[nc subgroups nn dist cobealg pdim]=scanparam(defopts,opts);
glabs=unique(group);
nfea=size(training,2);

%% extract feature
Ac=cell(1,numel(glabs));
for idx=1:numel(glabs)
    flag=group==glabs(idx);
    A=training(flag,:);
    
    %% split
    if subgroups>1
        trT=sum(flag);
        nCol=floor(trT/subgroups);
        n0=floor(trT/nCol);
        if trT-n0*nCol<nCol
            n0=n0-1;
        end
        split=[repmat(nCol,1,n0) trT-n0*nCol];
%         if min(split)>size(A,1)
%             error('Error specification of the number of subgroups.');
%         end
        if isempty(nc)
            cobe_opts.c=floor(min(split)*0.8);
        else
            cobe_opts.c=c;
        end
        
        switch lower(cobealg)
            case 'cobe'
                if isempty(pdim)
                    pdim=ceil(size(A,2)*.5);
                else
                    pdim=min(pdim,size(A,2));
                end
                cobe_opts.pdim=1000;
                [Ac{idx} Q]=pcobe(mat2cell(A',nfea,split),cobe_opts);
            case 'cobec'
                [Ac{idx} Q]=cobec(mat2cell(A',nfea,split),cobe_opts);
            case 'pcobe'
                [Ac{idx} Q]=cobec(mat2cell(A',nfea,split),cobe_opts);
            otherwise
                error('Unsupported algorithm.');
        end

        if nn
            [Ac{idx}]=cnfe(Ac{idx},Q);
        end       
    else    
        Ac{idx}=A';
    end
end


%% classifying
teT=size(testData,1);
labels=zeros(teT,1);

switch dist
    case 'correlation'  
        dis=zeros(numel(glabs),teT);
        %% testdata normalization
        testData=testData';
        testData=bsxfun(@minus,testData,mean(testData));
        testData=bsxfun(@rdivide,testData,max(sum(testData.*testData).^.5,eps));
        
        %% template normalization
        for idx=1:numel(glabs)
            Ac{idx}=bsxfun(@minus,Ac{idx},mean(Ac{idx}));
            Ac{idx}=orth(Ac{idx});
            proj=Ac{idx}'*testData;
            dis(idx,:)=sum(proj.^2).^.5;
        end
        [~,labels]=max(dis);
        labels=glabs(labels);  
        labels=labels(:);
    otherwise
        error('Unsupported distance.');
end
end

