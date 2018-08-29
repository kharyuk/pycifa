function MIhat = MutualInfo(L1,L2)
%   mutual information
% http://www.zjucadcg.cn/dengcai/Data/data.html
%  @ARTICLE{CHH05,
%         AUTHOR = {Deng Cai and Xiaofei He and Jiawei Han},
%         TITLE = {Document Clustering Using Locality Preserving Indexing},
%         JOURNAL = {IEEE Transactions on Knowledge and Data Engineering},
%         YEAR = {2005},
%         volume = {17},
%         number = {12},
%         pages = {1624-1637},
%         month = {December},}

%===========    
L1 = L1(:);
L2 = L2(:);
if size(L1) ~= size(L2)
    error('size(L1) must == size(L2)');
end

Label = unique(L1);
nClass = length(Label);

Label2 = unique(L2);
nClass2 = length(Label2);
if nClass2 < nClass
     % smooth
     L1 = [L1; Label];
     L2 = [L2; Label];
elseif nClass2 > nClass
     % smooth
     L1 = [L1; Label2];
     L2 = [L2; Label2];
end


G = zeros(nClass);
for i=1:nClass
    for j=1:nClass
        G(i,j) = sum(L1 == Label(i) & L2 == Label(j));
    end
end
sumG = sum(G(:));

P1 = sum(G,2);  P1 = P1/sumG;
P2 = sum(G,1);  P2 = P2/sumG;
if sum(P1==0) > 0 || sum(P2==0) > 0
    % smooth
    error('Smooth fail!');
else
    H1 = sum(-P1.*log2(P1));
    H2 = sum(-P2.*log2(P2));
    P12 = G/sumG;
    PPP = P12./repmat(P2,nClass,1)./repmat(P1,1,nClass);
    PPP(abs(PPP) < 1e-12) = 1;
    MI = sum(P12(:) .* log2(PPP(:)));
    MIhat = MI / max(H1,H2);
    %%%%%%%%%%%%%   why complex ?       %%%%%%%%
    MIhat = real(MIhat);
end



