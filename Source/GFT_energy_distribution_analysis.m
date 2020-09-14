%%  Statistical analysis over the action classes
% examine whether the energy distribution over GFT basis are correlated to
% action classes (on MSR3D dataset)

clear;

%% Determine the training and testing file listing and generate the corresponding label vectors
load('../LieGroup_combine_GFT/data/MSRAction3D/skeletal_data.mat');
load('../LieGroup_combine_GFT/data/MSRAction3D/body_model');

n_subjects = 10;
n_actions = 20;
n_instances = 3;

classNames = load('../LieGroup_combine_GFT/data/MSRAction3D/action_names.mat');

%% 
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
param.preProcessScheme = 3;

param.considerTSRatio = 1;

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
    At = At*cal_ratio_spatial_temporal_corr('MSRAction3D',param);
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

%% Energy distribution calculated for each sequence

n_sequences = length(find(skeletal_data_validity));        

features = cell(n_sequences, 1);
action_labels = zeros(n_sequences, 1);
subject_labels = zeros(n_sequences, 1);
instance_labels = zeros(n_sequences, 1); 

param.ratio_GFTcoeffs = 1;
param.numGFTcoeffs = ceil(param.numSpatialNode*param.numTempNode*param.ratio_GFTcoeffs);
param.signalGenMethod = 1;

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
                % apply the selected type of GFT 
                if param.typeofGFT==1
                    GFT_coeff = graphHist_GFT(skeletonData, Ast, opt_norm, param);
                elseif param.typeofGFT==2
                    GFT_coeff = graphHist_GFT_projector(skeletonData, Ast, opt_norm, param);
                end
                % calculate energy distribution over GFT basis (each dimension respectively)
                energyGFT = sum(GFT_coeff.^2, 1);
                energyDistb = energyGFT ./ sum(energyGFT);
                
                features{count} = energyDistb;
                action_labels(count) = action;       
                subject_labels(count) = subject;
                instance_labels(count) = instance;

                count = count + 1;
            end
        end
    end
end

%% Compare/analyze the energy distribution on GFT basis between action classes
% option1: generate one box plot among classes of interest for one GFT
% basis

addpath('plotting-utils/');

classInt = 1:20; % define the classes of interest for generating box plots
basisInt = 41:50; % define the basis index of interest for box plot
                 % 1-20:x-dim(horizontal); 21-40:y-dim(depth);
                 % 41-60:z-dim(vertical)
                 
dataPlot = cell2mat(features);
groupVar = action_labels;

% initialize 2D joint location
skeleton_loc = zeros(20,2);
skeleton_loc(20,:)=[0 3.6];
skeleton_loc(3,:)=[0 2.6]; 
skeleton_loc(1,:)=[-0.4 2.5];
skeleton_loc(2,:)=[0.4 2.5];
skeleton_loc(4,:)=[-0.2 2.2];
skeleton_loc(7,:)=[0 1];
skeleton_loc(5,:)=[-0.3 0.9];
skeleton_loc(6,:)=[0.3 0.9];
skeleton_loc(8,:)=[-0.6 1.2];
skeleton_loc(9,:)=[0.6 1.2];
skeleton_loc(10,:)=[-0.55 -0.05];
skeleton_loc(11,:)=[0.55 -0.05];
skeleton_loc(12,:)=[-0.45 -0.45];
skeleton_loc(13,:)=[0.45 -0.45];
skeleton_loc(14,:)=[-0.35 -0.5];
skeleton_loc(15,:)=[0.35 -0.5];
skeleton_loc(16,:)=[-0.4 -2.4];
skeleton_loc(17,:)=[0.4 -2.4];
skeleton_loc(18,:)=[-0.3 -2.8];
skeleton_loc(19,:)=[0.6 -2.7];

for basisIdx = basisInt
    figure;
    % boxplot plots one box for each column of x
    % data preparation for this box plot
    subplot('Position', [0.1 0.05 0.7 0.85]);
    boxplot(dataPlot(:,basisIdx), groupVar, 'Notch', 'on', 'Labels', classNames.action_names, 'LabelOrientation', 'inline');
    % xlabel('Action category');
    ylabel('Energy contribution ratio');
    title(sprintf('Box plot for feature index %02d', basisIdx));
    
    % plot the structure of corresponding basis vector
    subplot('Position', [0.82 0.5 0.15 0.3]);
    bIdx = mod(basisIdx,20);
    % first plot the fixed limbs
    for m=1:size(Es,2)
        line( [skeleton_loc(Es(1,m),1), skeleton_loc(Es(2,m),1)],  [skeleton_loc(Es(1,m),2), skeleton_loc(Es(2,m),2)],...
        'LineStyle', '--', 'Color','k','LineWidth',2);
    end
    % plot all the joints
    for m=1:20
        if V(m,bIdx)>10^(-5)
            line(skeleton_loc(m,1),skeleton_loc(m,2),'marker','o', 'markersize',6,'MarkerFaceColor','b','color', 'b');
        elseif V(m,bIdx)<(-1)*(10^(-5))
            line(skeleton_loc(m,1),skeleton_loc(m,2),'marker','o', 'markersize',6,'MarkerFaceColor','r','color', 'r');
        else
            line(skeleton_loc(m,1),skeleton_loc(m,2),'marker','o', 'markersize',6,'MarkerFaceColor','g','color', 'g');
        end
    end
    title(sprintf('%.03f',D(bIdx,bIdx)));
    axis([-1.5 1.5 -3 4]);
    axis off

end
                






