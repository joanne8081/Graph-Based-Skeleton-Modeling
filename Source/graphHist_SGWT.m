function SGWT_coeff = graphHist_SGWT(skeletonData, opSGWT, param)
% used for the normalizing skeleton data
% input: 
% SkeletonData: skeleton data (N x 3 x framesize) 
% opSGWT: SGWT transform operator matrix (NxC where C = (#scales+1)*#centerVertex)
% param.numGFTcoeffs: the number of GFT low-frequency coefficients to use
%                     in the feature vector
% param.numTempNode = 10;    % |Vt| the sliding temporal window size
% param.signalGenMethod = 2; % 1:raw skeleton position as graph signals; 
%                     2:signal as {p_i,t+j - p_ref,t}, i.e. relative positions
% if param.signalGenMethod==2 or 3
% param.refJointIdx: define the reference joint index

% output:
% coeffs: coefficients matrix (framesize x (3*C))

% if nargin < 4
%    ratio_coeffs = 1;
% end

[Ns, dim, T] = size(skeletonData);
Nt = param.numTempNode;
N = Ns*Nt;
assert(size(opSGWT,1)==N, 'Mismatch in size of the spatial temporal graph.');

% calculate the SGWT coefficient matrix
SGWT_coeff = zeros(T-Nt+1, size(opSGWT,2)*dim);
for frame=1:(T-Nt+1)
   % generate the graph signals
%    if param.signalGenMethod==1
    tempData = skeletonData(:,:,frame:(frame+Nt-1));
%    elseif param.signalGenMethod==2
%       tempData = skeletonData(:,:,frame:(frame+Nt-1)) - repmat(skeletonData(param.refJointIdx,:,frame), [Ns,1,Nt]);
%    elseif param.signalGenMethod==3
%       tempData = skeletonData(:,:,frame:(frame+Nt-1)) - repmat(skeletonData(param.refJointIdx,:,1), [Ns,1,Nt]);
%    end
   % generate the SGWT coefficients
   coef = [];
   for dim=1:3
      coef = [coef ; opSGWT' * reshape(tempData(:,dim,:),[],1)];
   end
   SGWT_coeff(frame,:) = coef';
   
end

end