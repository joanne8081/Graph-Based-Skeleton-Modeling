%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%% This is the script to evaluate spatial-temporal SGWT (with pre-determined spatial-temporal graph) 
%%% + temporal pyramid pooling (or Fourier temporal pyramid) features 
%%% on MSR Action3D Dataset
%%% Oct, 2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
tic;
%% Determine the training and testing file listing and generate the corresponding label vectors
load('../LieGroup_combine_GFT/data/MSRAction3D/skeletal_data.mat');
load('../LieGroup_combine_GFT/data/MSRAction3D/body_model');

load('../LieGroup_combine_GFT/data/MSRAction3D/skeletal_data.mat');
load('../LieGroup_combine_GFT/data/MSRAction3D/body_model');

n_subjects = 10;
n_actions = 20;
n_instances = 3;

classNames = load('../LieGroup_combine_GFT/data/MSRAction3D/action_names.mat');

%% Define parameters for the spatial-temporal graph

param.numSpatialNode = 20; % |Vs|
param.numTempNode = 1;    % |Vt|

%------------ specify SGWT parameters ------------
% kernelType = 'abspline3'; % other choices: 'mexican_hat', 'meyer', 'simple_tf'
kernelType = 'meyer';
numScales = 5; % specify the SGWT number of scales (M)
% scales = 3:10:43; % specify the SGWT scales to use (t_m) 
% can use sgwt_setscales function to set scales based on [lmin,lmax]
SGWTapprox = 0; % 0: compute wavelet by naive forward transform ; 1: compute wavelet by Chebyshev polynomial approximation
if SGWTapprox == 1
   approxOrder = 20; % specify the Chebyshev polynomial order (m)
end
% specify the list of centered vertices which to shift the scaling kernel to 
centerSpatialNode = 1:param.numSpatialNode;
centerTemporalNode = 1:param.numTempNode;
% centerSpatialNode = [3 7 8 9 14 15]; 
% centerTemporalNode = 1:floor(param.numTempNode/3):param.numTempNode;
centerVertex = repmat(centerSpatialNode, 1, length(centerTemporalNode)) + ...
   (reshape(repmat(centerTemporalNode, length(centerSpatialNode),1),[],1)' - ones(1, length(centerSpatialNode)*length(centerTemporalNode))) .* param.numSpatialNode;
%-------------------------------------------------

param.temporalMethod = 1; % 1:temporal pyramid pooling with max function ; 2:Fourier pyramid transform
param.maxLevel = 3; % TPM Level % pooling function: max functionm
if param.temporalMethod==2
   param.numTFFTcoeffs = 3; % specify the number of low-frequency coefficients to use per segment
end

param.preProcessScheme = 3;

param.classifyScheme = 1; % 1:linear SVM ; 2:RBF-SVM

param.considerTSRatio = 1;

param.STFT = 1;
if param.STFT==1
    param.STFTwindow = 8;
    param.STFTnumDC = 8;
end

%% Define the adjacency matrix for graph-based features

addpath('../Util/sgwt_toolbox/');
addpath('../Util/sgwt_toolbox/utils/');
addpath('../Util/sgwt_toolbox/mex/');
addpath('../Util/sgwt_toolbox/demo/');

% define spatial and temporal graphs edges
Es = [3 3 3 3  1 8  10 2 9  11 4 7 7 5  14 16 6  15 17;
         1 2 4 20 8 10 12 9 11 13 7 5 6 14 16 18 15 17 19];   
% Es = [3 3 3 3  1 8  2 9  4 7 7 5  14 6  15;
%       1 2 4 20 8 10 9 11 7 5 6 14 16 15 17]; % disconnect the 4 limb-end joints   
   
Et = [1:param.numTempNode-1 ; 2:param.numTempNode];

% calculate the adjacency matrices for two graphs
As = zeros(param.numSpatialNode , param.numSpatialNode);
for n=1:size(Es,2)
   As(Es(1,n), Es(2,n)) = 1;
   As(Es(2,n), Es(1,n)) = 1;
end
At = zeros(param.numTempNode, param.numTempNode);
for n=1:size(Et,2)
   At(Et(1,n), Et(2,n)) = 1;
   At(Et(2,n), Et(1,n)) = 1;
end

% Consider to scale temporal edge weight based on TS correlation ratio
if param.considerTSRatio==1
    At = At*cal_ratio_spatial_temporal_corr('MSRAction3D',param);
end

% the Cartesian product of node i in spatial graph and node j in temporal
% graph will be the (i,j) node in S-T graph with node index = (j-1)*|Vs|+i
Ast = kron(eye(param.numTempNode),As) + kron(At,eye(param.numSpatialNode));
Lst = diag(Ast*ones(size(Ast,2),1))-Ast;
[V,D]=eig(Lst);
lmax = D(end,end); % find the maximum spectrum range

% specify and compute the kernels to use in SGWT
[g,sgwt_t]=sgwt_filter_design(lmax, numScales, 'designtype', kernelType);
arange=[0 lmax];
% compute the transform operator matrix SGWT_T
opSGWT = zeros(size(Lst,1), (numScales+1)*length(centerVertex));
for scaleIdx = 1:numScales+1
   for n = 1:length(centerVertex)
      d = sgwt_delta(size(Lst,1), centerVertex(n)); % generating delta function localized at selected vertices
      r = sgwt_ftsd(d,g{1,scaleIdx},1,Lst); % convolve delta function with scaled kernel function on graph 
      opSGWT(:, (scaleIdx-1)*length(centerVertex)+n) = r;
   end
end

%% Plot the filter function versus the graph spectrum 
figure;
stem(diag(D), 1.2*ones(length(diag(D)),1), ':ok','fill');
hold on
sampPts = 0:0.01:D(end,end);
% plot(sampPts, g{1}(sampPts), '-b');
% plot(sampPts, g{2}(sampPts), '-g');
% plot(sampPts, g{3}(sampPts), '-r');
% plot(sampPts, g{4}(sampPts), '-c');
% plot(sampPts, g{5}(sampPts), '-y');
% plot(sampPts, g{6}(sampPts), '-m');
for scaleIdx = 1:numScales+1
    plot(sampPts, g{scaleIdx}(sampPts), '-b');
end
xlabel('\lambda');
axis([0 D(end,end) 0 1.4]);
title(sprintf('#TempNode=%d; #Scales=%d; Kernel=%s', param.numTempNode, numScales, kernelType));


%% Feature generating

n_sequences = length(find(skeletal_data_validity));        

features = cell(n_sequences, 1);
action_labels = zeros(n_sequences, 1);
subject_labels = zeros(n_sequences, 1);
instance_labels = zeros(n_sequences, 1); 

count = 1;
for subject = 1:n_subjects
    for action = 1:n_actions
        for instance = 1:n_instances
            if (skeletal_data_validity(action, subject, instance))                    
                
                joint_locations = skeletal_data{action, subject, instance}.joint_locations;     % dim ~ 3x20xT   
                S = size(joint_locations);
                % Wrap to match the graph index (resize the skeletonData as a matrix of Nx3xT)              
                skeletonData = zeros(S(2),S(1),S(3));
                for t = 1:S(3)
                    skeletonData(:,:,t) = joint_locations(:,:,t)';
                end
                if param.preProcessScheme == 0
                elseif param.preProcessScheme == 1
                    skeletonData = diff(skeletonData,1,3);     
                    skeleton_v = skeletonData;
                    skeletonData =  skeleton_v([1:6 8:20], :, :);
                    S(2) = S(2)-1;
                elseif param.preProcessScheme == 2
                    %%%%%%%%%%%% TEST ADD HIP COORD BACK %%%%%%%%%%%
                    ORI = skeletal_data{action, subject, instance}.original_skeletal_data;
                    hipLocation = ORI(:,7,:);
                    hipLocation = reshape(hipLocation,[1 S(1) S(3)]);
                    hipLocation = repmat(hipLocation, [S(2) 1 1]);
                    skeletonData = skeletonData + hipLocation;
                    skeletonData = diff(skeletonData,1,3);
                    S(3) = S(3)-1;
                    %%%%%%%%%%%% TEST ADD HIP COORD BACK %%%%%%%%%%%
                elseif param.preProcessScheme == 3
                    joint_locations = skeletal_data{action, subject, instance}.original_skeletal_data;
                    S = size(joint_locations);
                    % Wrap to match the graph index (resize the skeletonData as a matrix of Nx3xT)              
                    skeletonData = zeros(S(2),S(1),S(3));
                    for t = 1:S(3)
                        skeletonData(:,:,t) = joint_locations(:,:,t)';
                    end
                    % transit the hip location in first frame to the origin
                    temp = repmat(skeletonData(7,:,1), [S(2),1,S(3)]);
                    skeletonData = skeletonData - temp;
                elseif param.preProcessScheme == 4
                    joint_locations = skeletal_data{action, subject, instance}.original_skeletal_data;
                    S = size(joint_locations);
                    % Wrap to match the graph index (resize the skeletonData as a matrix of Nx3xT)              
                    skeletonData = zeros(S(2),S(1),S(3));
                    for t = 1:S(3)
                        skeletonData(:,:,t) = joint_locations(:,:,t)';
                    end
                    % transit the hip location in first frame to the origin
                    temp = repmat(skeletonData(7,:,1), [S(2),1,S(3)]);
                    skeletonData = skeletonData - temp;
                    skeletonData = diff(skeletonData,1,3);
                    S(3) = S(3)-1;
                end
                if S(3)<param.numTempNode
                    for t = S(3)+1 : param.numTempNode
                        skeletonData(:,:,t) = joint_locations(:,:,S(3))';
                    end
                end
                SGWT_coeff = graphHist_SGWT(skeletonData, opSGWT, param);
                % TEST: Short-time Fourier Transform
                if param.STFT==1
                    newSGWT_coeff = zeros((size(SGWT_coeff,1)-param.STFTwindow+1), param.STFTnumDC*size(SGWT_coeff,2));
                    for t=1:(size(SGWT_coeff,1)-param.STFTwindow+1)
                        tempBlk = SGWT_coeff(t:t+param.STFTwindow-1, :);
                        tempFFT = abs(fft(tempBlk));
                        newSGWT_coeff(t, :) = reshape(tempFFT(1:param.STFTnumDC,:), 1, param.STFTnumDC*size(SGWT_coeff,2));
                    end
                end
                
                if param.temporalMethod==1
                    if param.STFT==1
                        [SGWTfeatures] = pyramid_pool(newSGWT_coeff, param.maxLevel, 1); %0:mean, 1:max function
                    elseif param.STFT==0
                        [SGWTfeatures] = pyramid_pool(SGWT_coeff, param.maxLevel, 1); %0:mean, 1:max function
                    end
                elseif param.temporalMethod==2
                    [SGWTfeatures] = fft_pyramid(SGWT_coeff, param.maxLevel, param.numTFFTcoeffs);
                end
                features{count} = SGWTfeatures;
                action_labels(count) = action;       
                subject_labels(count) = subject;
                instance_labels(count) = instance;

                count = count + 1;
            end
        end
    end
end

%mkdir('UTKinect/ST_SGWT');
%save('UTKinect/ST_SGWT/features', 'features');
%save('UTKinect/ST_SGWT/labels', 'action_labels', 'subject_labels', 'instance_labels');

%% Classification

load('../LieGroup_combine_GFT/data/MSRAction3D/tr_te_splits.mat');
accuracyFold = zeros(size(tr_subjects,1),1);
aggreCMat = zeros(n_actions, n_actions);

adoptPCA = 1; % 0: not adopting PCA

for fold=1:size(tr_subjects,1)
    train_subList = tr_subjects(fold,:);
    test_subList = te_subjects(fold,:);
% train_subList = [1 3 5 7 9]; % subjects for training.
% test_subList = [2 4 6 8 10]; % subjects for testing.

% all_subList = 1:10;
% all_actList = 1:10;

train_FeatureSet = cell2mat(features(ismember(subject_labels, train_subList), 1));
train_class_labs = action_labels(ismember(subject_labels, train_subList), 1);
test_FeatureSet = cell2mat(features(ismember(subject_labels, test_subList), 1));
test_class_labs = action_labels(ismember(subject_labels, test_subList), 1);

%%%%%%%%%%%%%%%
if adoptPCA==1
    [COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(train_FeatureSet);
    cumdist = cumsum(EXPLAINED);
    numCP = min(find(cumdist >= 99));
    train_FeatureSet = SCORE(:, 1:end);
    temp = test_FeatureSet - repmat(MU, size(test_FeatureSet,1), 1);
    test_FeatureSet = temp*COEFF(:, 1:end);
end

%%%%%%%%%%%%%%%

addpath('../Util/libsvm-3.18/matlab');

if param.classifyScheme==1
   model = svmtrain(train_class_labs, train_FeatureSet, '-t 0');
   [predict_label, test_accuracy, dec_values] = svmpredict(test_class_labs, test_FeatureSet, model); 
   accuracyFold(fold) = test_accuracy(1);

elseif param.classifyScheme==2
   % determine hyperparameters for RBF-SVM with cross-validation
   cVal=-1:11;
   cVal = 10.^cVal;
   gammaVal = -8:3;
   gammaVal = 10.^gammaVal;
   avgAccu = zeros(length(cVal), length(gammaVal)); 

   for m=1:length(cVal)
      for n=1:length(gammaVal)
         s = ['-c ', num2str(cVal(m)), ' -t 2 -g ',num2str(gammaVal(n)), ' -v 5 -q'];
         model = svmtrain(train_class_labs, train_FeatureSet, s);
         avgAccu(m,n) = model;
      end
   end
   [Y1,I1]=max(avgAccu,[],1);
   [Y2,I2]=max(Y1); 
   bestGamma = gammaVal(I2);
   bestC = cVal(I1(I2));
   s = ['-c ', num2str(bestC), ' -t 2 -g ',num2str(bestGamma), ' -q'];
   % train on the total training dataset
   model = svmtrain(train_class_labs, train_FeatureSet, s);
   [predict_label, test_accuracy, dec_values] = svmpredict(test_class_labs, test_FeatureSet, model); 
%    accuracyFold(fold) = test_accuracy(1);
   
end

cmat = confusionmat(test_class_labs, predict_label);
cmat = cmat ./ repmat(sum(cmat,2), 1, size(cmat,2));
aggreCMat = aggreCMat + cmat;

end
% plotConfMat(cmat, classNames.action_names);
addpath('plotting-utils');
plotConfMat(aggreCMat/size(tr_subjects,1), classNames.action_names,1);
title('Confusion matrix with ST-SGWT+TPM (1-vs-1) on MSR-Action3D')
% end
toc;


