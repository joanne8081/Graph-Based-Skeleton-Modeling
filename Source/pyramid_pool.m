function [features] = pyramid_pool(Coeff, level, func_type)
% pyramid pooling function
% input:
% Coeff: TxD coefficient matrix
% level: number of pyramid levels, level >=1
% func_type: summary function to be used. (0: mean pooling; 1: max
% pooling; 2: sum pooling)
% output:
% features: 1xN feature vector
[T, D] = size(Coeff);
features = [];

% make sure T >= maximum number of subblocks or padding the coefficients
% assert(T>=2^(level-1),'length of sequence is too short for splitting sub-blocks ');
if T<2^(level-1)
    Coeff = [Coeff ; repmat(Coeff(T,:),(2^(level-1)-T),1)];
end

for k=1:level
    numBlocks = 2^(k-1);
    numRows = floor(size(Coeff,1)/numBlocks);
    zk = zeros(1, (D*numBlocks));
    for b=1:numBlocks
        if func_type==0
           tempBlk = Coeff(((b-1)*numRows+1) : min(size(Coeff,1), b*numRows) , : );
           zk(((b-1)*D+1) : b*D) = mean(tempBlk,1);
        elseif func_type==1
           tempBlk = Coeff(((b-1)*numRows+1) : min(size(Coeff,1), b*numRows) , : );
           zk(((b-1)*D+1) : b*D) = max(tempBlk,[],1);
        elseif func_type==2
           tempBlk = Coeff(((b-1)*numRows+1) : min(size(Coeff,1), b*numRows) , : );
           zk(((b-1)*D+1) : b*D) = sum(tempBlk,1);
        end
    end
    features = [features, zk];
end

end