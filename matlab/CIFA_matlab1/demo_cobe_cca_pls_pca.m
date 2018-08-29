clear;
close all;
clc;

N=2;
rC=1;
rI=9;
M=rI+rC;
T=1000;
s=[sin(.01*[1:T])' sign(sin(.01*[1:T]))'];

A{1}=s(:,1);
A{2}=s(:,2);

A{1}(:,2:rI+1)=randn(1000,rI);
A{2}(:,2:rI+1)=randn(1000,rI);
A{1}=A{1}*randn(rI+1,rI+1);
A{2}=A{2}*randn(rI+1,rI+1);

opts.c=1;
c=cobe(A,opts);
px=A{1}*(A{1}\c);
py=A{2}*(A{2}\c);

[xl yl xs ys]=plsregress(A{1},A{2},rC);
xpls=datanormalize(xs(:,1));
ypls=datanormalize(ys(:,1));


[wa wb r xl yl]=canoncorr(A{1},A{2});

xcca=datanormalize((A{1}*wa(:,1:rC)));
ycca=datanormalize((A{2}*wb(:,1:rC)));


Amat=[A{:}];
[pca d v]=svds(Amat,1);

At=cellfun(@(x) x',A,'uni',false);
[JAc JAi]=JIVE(At,rC,[rI rI],'y',10^(-10),5000);
[temp d JAc]=svds(JAc,rC,'L');


figure('units','normalized','outerposition',[0.2 0.55 0.6 0.4],'Name','COBE vs. CCA');
subplot(211);
plot([px py c]);
axis tight;
xlabel('COBE')
legend({'$\mathbf{Y}_1\mathbf{w}_1$','$\mathbf{Y}_2\mathbf{w}_2$','$\mathbf{\bar{a}}_1$'},'Interpreter', 'latex')

subplot(212);
plot([xcca,ycca]);
axis tight;
xlabel('Canonical Correlation Analysis');
legend({'$\mathbf{Y}_1\mathbf{w}_1$','$\mathbf{Y}_2\mathbf{w}_2$'},'Interpreter','latex')

figure('units','normalized','outerposition',[0.2 0.3 0.6 0.25],'Name','COBE vs. PLS');
sig=sign(corr(xpls,c));
plot([xpls ypls sig*c]);
legend({'PLS-$\mathbf{Y}_1$','PLS-$\mathbf{Y}_2$','$\mathbf{\bar{a}}_1$'},'Interpreter','latex');
axis tight;


figure('units','normalized','outerposition',[0.2 0.05 0.6 0.25],'Name','COBE vs. PCA and JIVE');
sig=sign(corr(pca,c));
sigJ=sign(corr(pca,JAc));
plot([pca sigJ*JAc sig*c]);
ls=get(gca,'child');
set(ls(1),'color','red','linewidth',2);
legend({'PCA','JIVE','COBE'},'Interpreter','latex');
axis tight;
