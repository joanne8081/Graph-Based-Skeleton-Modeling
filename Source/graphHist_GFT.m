function GFT_coeff = graphHist_GFT(skeletonData, Ast, opt_norm, param)
% used for the normalizing skeleton data
% input: 
% SkeletonData: skeleton data (N x 3 x framesize) 
% Ast: adjacency matrix for the predefined spatial-temporal graph
% opt_norm: option for Laplacian matrix (0: unnormalized L, 1: normalized L)
% param.numGFTcoeffs: the number of GFT low-frequency coefficients to use
%                     in the feature vector
% param.numTempNode = 10;    % |Vt| the sliding temporal window size
% param.signalGenMethod = 2; % 1:raw skeleton position as graph signals; 
%                     2:signal as {p_i,t+j - p_ref,t}, i.e. relative positions
% if param.signalGenMethod==2 or 3
% param.refJointIdx: define the reference joint index

% output:
% coeffs: coefficients matrix (framesize x (3*numGFTcoeffs))

% if nargin < 4
%    ratio_coeffs = 1;
% end

[Ns, dim, T] = size(skeletonData);
Nt = param.numTempNode;
N = Ns*Nt;
assert(size(Ast,1)==N, 'Mismatch in size of the spatial temporal graph.');

% GFT coefficients
GFT_coeff = zeros(T, param.numGFTcoeffs*dim);

% get Degree Matrix
deg = diag(Ast*ones(N,1));
if opt_norm==0
   % get Laplace Matrix
   lap = deg - Ast;
   [n_V, n_D] = eig(lap);
elseif opt_norm==1
   % get normalized Laplace Matrix
   n_lap = eye(N) - (deg^(-0.5))*Ast*(deg^(-0.5));
   [n_V, n_D] = eig(n_lap);
end

% calculate the coefficient matrix
GFT_coeff = zeros(T-Nt+1, param.numGFTcoeffs*dim);
for frame=1:(T-Nt+1)
   % generate the graph signals
   if param.signalGenMethod==1
      tempData = skeletonData(:,:,frame:(frame+Nt-1));
   elseif param.signalGenMethod==2
      tempData = skeletonData(:,:,frame:(frame+Nt-1)) - repmat(skeletonData(param.refJointIdx,:,frame), [Ns,1,Nt]);
   elseif param.signalGenMethod==3
      tempData = skeletonData(:,:,frame:(frame+Nt-1)) - repmat(skeletonData(param.refJointIdx,:,1), [Ns,1,Nt]);
   end
   % generate the GFT coefficients
   coef = [];
   for dim=1:3
      coef = [ coef ; (n_V(:,1:param.numGFTcoeffs)' * reshape(tempData(:,dim,:), [], 1))];
   end
   GFT_coeff(frame,:) = coef';
end

end