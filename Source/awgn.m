function Y = awgn(X, SNR, seed)
% Add white Gaussian noise to the input signal vector/matrix X to generage
%%%%% Input %%%%%
% X: original signal vector/matrix with size NxD (each row is one sample)
% SNR: target SNR in dB
% seed: (opt)seed for random number generator
%%%%% Output %%%%%
% Y: resulting signal vector/matrix with noise added of the same size as X
%%%%%
% Author: Jiun-Yu Joanne Kao
% Date: Sep. 2017

if nargin < 3
    seed = now;
end
rng(seed);
[N,D] = size(X);
SNR_lin = 10^(SNR/10);
% noise = zeros(size(X));
% for n=1:N
%     E_sig = sum(X(n,:).^2)/D;
%     E_n = E_sig / SNR_lin;
%     sigma = sqrt(E_n);
%     noise(n,:) = sigma * randn(1,D);
% end
% change to generate the noise based on energy of signal of whole seq
E_sig = sum(sum(X.^2))/(N*D);
E_n = E_sig / SNR_lin;
sigma = sqrt(E_n);
noise = sigma * randn(N,D);
Y = X + noise;

end