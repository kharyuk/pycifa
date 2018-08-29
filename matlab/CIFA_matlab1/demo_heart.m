%% demo for simulation 2
clear;
clc;
load heart;
S=S';
N=4;

rC=2;
rI=8;
nMix=20;

A=cell(1,N);
for n=1:N
    A{n}(:,1:rC)=(S(:,1:rC));
    A{n}(:,rC+1:rC+rI)=(rand(size(S,1),rI));
    
    %% mixing
    A{n}=A{n}*(rand(rC+rI,nMix));    
end

opts.c=rC;
[c Q]=cobe(A,opts);  % common features extraction
nopts.NumOfComp=rC;
[nc nQ]=cnfe(c,Q,nopts); % nonnegative common features extraction


%% visualization of rersults.
B=cellfun(@(x) x(:,1),A,'uni',false);
fmix=figure('name','Four observations','units','normalized','visible','off');
visual([B{:}],1,4,164,true);
movegui(fmix,'center');
pos_fmix=get(fmix,'outerposition');
pos_fmix(2)=pos_fmix(2)+pos_fmix(4)/2;
set(fmix,'outerposition',pos_fmix);
set(fmix,'visible','on');


figure('name','Recovered signals','units','normalized','outerposition',[pos_fmix(1) pos_fmix(2)-pos_fmix(4) pos_fmix(3)/2 pos_fmix(4)]);
nc=datanormalize(nc,inf)*255;
visual(nc,1,rC,164,true);
figure('name','Source signals','units','normalized','outerposition',[pos_fmix(1)+pos_fmix(3)/2 pos_fmix(2)-pos_fmix(4) pos_fmix(3)/2 pos_fmix(4)]);
visual(S,1,rC,164,true);