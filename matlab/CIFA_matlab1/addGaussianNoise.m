function [ sn ] = addGaussianNoise( s, SNR )
%% This function is used to add i.i.d. Gaussian noise to the rows of s when 'awgn' is unavailable.
[R T]=size(s);
if nargin==1
    SNR=20;
end
sn=zeros(R,T);
for r=1:R
    powS=sum(s(r,:).^2)^.5;
    noi=randn(1,T);
    powN=sum(noi.^2)^.5;
    p=10^(-SNR/20)*powS/powN;
    sn(r,:)=s(r,:)+p*noi;
end

