function [features] = fft_pyramid(Coeff, level, num_coeffs)
% pyramid pooling function
% input:
% Coeff: TxD coefficient matrix
% level: number of pyramid levels, level >=1
% num_coeffs: the number of low-frequency coefficients in each segment to use as features (default: 5)
% 
% output:
% features: 1xN feature vector

if nargin < 3
   num_coeffs = 5;
end

[T, D] = size(Coeff);
features = [];

for k=1:level
    numBlocks = 2^(k-1);
    numRows = ceil(T/numBlocks);
    zk = zeros(1, (num_coeffs*D*numBlocks));
    for b=1:numBlocks
       tempBlk = Coeff(((b-1)*numRows+1) : min(T, b*numRows) , : );
       tempFFT = abs(fft(tempBlk));
       if size(tempFFT,1)>=num_coeffs
          zk((((b-1)*num_coeffs*D)+1) : b*num_coeffs*D) = reshape(tempFFT(1:num_coeffs,:), 1, num_coeffs*D);
       else
          zk((((b-1)*num_coeffs*D)+1) : b*num_coeffs*D) = zeros(1,num_coeffs*D);
       end
    end
    features = [features, zk];
end

end