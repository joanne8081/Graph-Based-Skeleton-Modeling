%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%% Experiment on comparing the effect of adding Gaussian noises to data 
%%% on the performance of representation between graph-based method and
%%% PCA-based method
%%% on UTKinect dataset
%%% Oct. 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Determine the training and testing file listing and generate the corresponding label vectors
load('../LieGroup_combine_GFT/data/UTKinect/skeletal_data');
load('../LieGroup_combine_GFT/data/UTKinect/body_model');
addpath('../Util/libsvm-3.18/matlab');

n_subjects = 10;
n_actions = 10;
n_instances = 2;

%% Define parameters for the spatial-temporal graph

param.numSpatialNode = 20; % |Vs|
param.numTempNode = 1;    % |Vt|

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

param.considerTSRatio = 0;

param.typeofGFT = 1; % 1:conventional GFT ; 2:spectral projector-based GFT
opt_norm = 1;  % option for Laplacian matrix (0: unnormalized L, 1: normalized L)

%% Define the adjacency matrix for graph-based features

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
    At = At*cal_ratio_spatial_temporal_corr('UTKinect',param);
end

% the Cartesian product of node i in spatial graph and node j in temporal
% graph will be the (i,j) node in S-T graph with node index = (j-1)*|Vs|+i
Ast = kron(eye(param.numTempNode),As) + kron(At,eye(param.numSpatialNode));
% get Degree Matrix
deg = diag(Ast*ones(size(Ast,1),1));
if opt_norm==0
   % get Laplace Matrix
   lap = deg - Ast;
   [V, D] = eig(lap);
elseif opt_norm==1
   % get normalized Laplace Matrix
   n_lap = eye(size(Ast,1)) - (deg^(-0.5))*Ast*(deg^(-0.5));
   [V, D] = eig(n_lap);
end

%% Add Gaussian noises of different level to the data AND saved the corrupted data as a cell array

numExp = 10;
snrVal = [-50:5:60]; % in dB
% snrVal = 0;
train_subList = [1 3 5 7 9]; % subjects for training
test_subList = [2 4 6 8 10]; % subjects for testing
accuracy_pca = zeros(numExp, length(snrVal));
accuracy_gft = zeros(numExp, length(snrVal));

% Calculate E_sig across all training data for generating wgn
skeletonALL = [];
for subject = 1:train_subList
    for action = 1:n_actions
        for instance = 1:n_instances
            if (skeletal_data_validity(action, subject, instance)) 
                joint_locations = skeletal_data{action, subject, instance}.original_skeletal_data*100;     % dim ~ 3x20xT   
                S = size(joint_locations);
                % wrap to match the graph index (resize the skeletonData as a matrix of Nx3xT)              
                skeletonData = zeros(S(2),S(1),S(3));
                for t = 1:S(3)
                    skeletonData(:,:,t) = joint_locations(:,:,t)';
                end
                skeletonALL = cat(3, skeletonALL, skeletonData);
            end
        end
    end
end
% rng(now);
% [N,D] = size(X);
% SNR_lin = 10^(SNR/10);
E_sig = sum(sum(sum(skeletonALL.^2)))/(size(skeletonALL,1)*size(skeletonALL,2)*size(skeletonALL,3));
% E_n = E_sig / SNR_lin;
% sigma = sqrt(E_n);
% noise = sigma * randn(N,D);
% Y = X + noise;

%%

for expIdx=1:numExp

% parameters for GFT
param.ratio_GFTcoeffs = 1;
param.numGFTcoeffs = ceil(param.numSpatialNode*param.numTempNode*param.ratio_GFTcoeffs);
param.signalGenMethod = 1;

for snrIdx = 1:length(snrVal)
    % Generate corrupted skeletal data with the given SNR and also
    % preprocessed it
    snr = snrVal(snrIdx);
    SNR_lin = 10^(snr/10);
    E_n = E_sig / SNR_lin;
    sigma = sqrt(E_n);
    rng(now);
    
    corrupt_skeletal_data = cell(n_actions, n_subjects, n_instances);
    for subject = 1:n_subjects
        for action = 1:n_actions
            for instance = 1:n_instances
                if (skeletal_data_validity(action, subject, instance)) 
                    joint_locations = skeletal_data{action, subject, instance}.original_skeletal_data*100;     % dim ~ 3x20xT   
                    S = size(joint_locations);
                    % wrap to match the graph index (resize the skeletonData as a matrix of Nx3xT)              
                    skeletonData = zeros(S(2),S(1),S(3));
                    corrupt_skeletonData = zeros(S(2),S(1),S(3));
                    for t = 1:S(3)
                        skeletonData(:,:,t) = joint_locations(:,:,t)';
%                         corrupt_skeletonData(:,:,t) = awgn(skeletonData(:,:,t), snr);
                        corrupt_skeletonData(:,:,t) = skeletonData(:,:,t) + (sigma * randn(S(2),S(1)));
                    end
                    % transit the hip location in first frame to the origin
                    temp = repmat(corrupt_skeletonData(7,:,1), [S(2),1,S(3)]);
                    corrupt_skeletonData = corrupt_skeletonData - temp;
                    corrupt_skeletal_data{action, subject, instance} = corrupt_skeletonData;
                end
            end
        end
    end
    % With the manually generated noisy data, preprocess and find PCA basis
    corrupt_dataMAT = [];
    for subject = train_subList
        for action = 1:n_actions
            for instance = 1:n_instances
                if (skeletal_data_validity(action, subject, instance)) 
                    corrupt_dataMAT = [corrupt_dataMAT, reshape(corrupt_skeletal_data{action, subject, instance}, 20, [])];
                end
            end
        end
    end
    [COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(corrupt_dataMAT');
    
    % For each sequence, generate two feature vectors, one with PCA and
    % the other with graph-based method
    % using the same temporal pooling scheme
    n_sequences = length(find(skeletal_data_validity));
    features_pca = cell(n_sequences, 1);
    features_gft = cell(n_sequences, 1);
    action_labels = zeros(n_sequences, 1);
    subject_labels = zeros(n_sequences, 1);
    instance_labels = zeros(n_sequences, 1); 
    count = 1;
    for subject = 1:n_subjects
        for action = 1:n_actions
            for instance = 1:n_instances
                if (skeletal_data_validity(action, subject, instance))
                    skData = corrupt_skeletal_data{action, subject, instance};
                    % Apply PCA 
                    PCA_coeff = zeros(size(skData,3), size(skData,1)*size(skData,2));
                    for t=1:size(skData,3)
                        temp = COEFF' * (skData(:,:,t)-repmat(MU',1,3));
                        temp = reshape(temp, [], 1);
                        PCA_coeff(t,:) = temp';
                    end
                    % Apply GFT
                    GFT_coeff = graphHist_GFT(skData, Ast, opt_norm, param);
                    % Apply same selected type of temporal modeling to both
                    % pca and gft
                    if param.temporalMethod==1
                        [PCAfeatures] = pyramid_pool(PCA_coeff, param.maxLevel, 1);
                        [GFTfeatures] = pyramid_pool(GFT_coeff, param.maxLevel, 1); %0:mean, 1:max function
                    elseif param.temporalMethod==2
                        [PCAfeatures] = fft_pyramid(PCA_coeff, param.maxLevel, param.numTFFTcoeffs);
                        [GFTfeatures] = fft_pyramid(GFT_coeff, param.maxLevel, param.numTFFTcoeffs);
                    end
                    features_pca{count} = PCAfeatures;
                    features_gft{count} = GFTfeatures;
                    action_labels(count) = action;       
                    subject_labels(count) = subject;
                    instance_labels(count) = instance;
                    count = count + 1;
                end
            end
        end
    end
    
    % Classification
    train_FeatureSet_pca = cell2mat(features_pca(ismember(subject_labels, train_subList), 1));
    test_FeatureSet_pca = cell2mat(features_pca(ismember(subject_labels, test_subList), 1));
    train_FeatureSet_gft = cell2mat(features_gft(ismember(subject_labels, train_subList), 1));
    test_FeatureSet_gft = cell2mat(features_gft(ismember(subject_labels, test_subList), 1));
    train_class_labs = action_labels(ismember(subject_labels, train_subList), 1);
    test_class_labs = action_labels(ismember(subject_labels, test_subList), 1);
    % result for PCA
    model = svmtrain(train_class_labs, train_FeatureSet_pca, '-t 0');
    [predict_label, test_accuracy, dec_values] = svmpredict(test_class_labs, test_FeatureSet_pca, model); 
    accuracy_pca(expIdx, snrIdx) = test_accuracy(1);
    % result for GFT
    model = svmtrain(train_class_labs, train_FeatureSet_gft, '-t 0');
    [predict_label, test_accuracy, dec_values] = svmpredict(test_class_labs, test_FeatureSet_gft, model); 
    accuracy_gft(expIdx, snrIdx) = test_accuracy(1);
end

end

%% Plot the results
figure;
plot(snrVal, mean(accuracy_pca,1), 'ro-','LineWidth',3.5,'MarkerEdgeColor','k','MarkerFaceColor','k',...
                       'MarkerSize',7);
hold on
plot(snrVal, mean(accuracy_gft,1), 'bx-','LineWidth',3.5,'MarkerEdgeColor','k','MarkerFaceColor','k',...
                       'MarkerSize',15);
legend({'PCA', 'GFT'}, 'fontsize', 22, 'Location', 'SouthEast');
% axis([-10 70 65 75])
xlabel('SNR (dB)', 'fontsize', 22);
ylabel('Classification accuracy (%)', 'fontsize', 22);
title('Classification accuracy over different SNR levels', 'fontsize', 22);






