% calculate the ratio between spatial correlation and temporal correlation

function TtoS_ratio = cal_ratio_spatial_temporal_corr(dataset,param)

if (strcmp(dataset, 'MSRAction3D'))
    
    load('../LieGroup_combine_GFT/data/MSRAction3D/skeletal_data.mat');
    load('../LieGroup_combine_GFT/data/MSRAction3D/body_model');

    n_subjects = 10;
    n_actions = 20;
    n_instances = 3;

    Es = [3 3 3 3  1 8  10 2 9  11 4 7 7 5  14 16 6  15 17;
             1 2 4 20 8 10 12 9 11 13 7 5 6 14 16 18 15 17 19];   
    % Es = [3 3 3 3  1 8  2 9  4 7 7 5  14 6  15;
    %       1 2 4 20 8 10 9 11 7 5 6 14 16 15 17]; % disconnect the 4 limb-end joints   

    Et = [1:param.numTempNode-1 ; 2:param.numTempNode];

    n_sequences = length(find(skeletal_data_validity));        
    n_joints = 20;

    spatialCov = zeros(size(Es,2),1);
    temporalCov = zeros(n_joints,1);

    count = 0;
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
                    % calculate pairwise spatial cov
                    for k = 1:size(Es,2)
                        temp = 0;
                        for dim = 1:3
                            C = corrcoef(reshape(skeletonData(Es(1,k),dim,:),[],1), reshape(skeletonData(Es(2,k),dim,:),[],1));
                            temp = temp + abs(C(1,2));
                        end
                        spatialCov(k) = spatialCov(k)+temp;
                    end
                    % calculate temporal cov
                    for k = 1:n_joints
                        temp = 0;
                        for dim = 1:3
                            C = corrcoef(reshape(skeletonData(k,dim,1:(end-1)),[],1), reshape(skeletonData(k,dim, 2:end),[],1));
                            temp = temp + abs(C(1,2));
                        end
                        temporalCov(k) = temporalCov(k)+temp;
                    end

                    count = count + 1;

                end
            end
        end
    end

    temp = spatialCov(~isnan(spatialCov));
    spatialCorrM = mean(temp);
    temp = temporalCov(~isnan(temporalCov));
    temporalCorrM = mean(temp);
    TtoS_ratio = temporalCorrM / spatialCorrM 
    % 1.0757 

elseif (strcmp(dataset, 'UTKinect'))
    
    load('../LieGroup_combine_GFT/data/UTKinect/skeletal_data');
    load('../LieGroup_combine_GFT/data/UTKinect/body_model');

    n_subjects = 10;
    n_actions = 10;
    n_instances = 2;
    
    Es = [3 3 3 3  1 8  10 2 9  11 4 7 7 5  14 16 6  15 17;
             1 2 4 20 8 10 12 9 11 13 7 5 6 14 16 18 15 17 19];   
    % Es = [3 3 3 3  1 8  2 9  4 7 7 5  14 6  15;
    %       1 2 4 20 8 10 9 11 7 5 6 14 16 15 17]; % disconnect the 4 limb-end joints   

    Et = [1:param.numTempNode-1 ; 2:param.numTempNode];

    n_sequences = length(find(skeletal_data_validity));        
    n_joints = 20;

    spatialCov = zeros(size(Es,2),1);
    temporalCov = zeros(n_joints,1);

    count = 0;
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
                    % calculate pairwise spatial cov
                    for k = 1:size(Es,2)
                        temp = 0;
                        for dim = 1:3
                            C = corrcoef(reshape(skeletonData(Es(1,k),dim,:),[],1), reshape(skeletonData(Es(2,k),dim,:),[],1));
                            temp = temp + abs(C(1,2));
                        end
                        spatialCov(k) = spatialCov(k)+temp;
                    end
                    % calculate temporal cov
                    for k = 1:n_joints
                        temp = 0;
                        for dim = 1:3
                            C = corrcoef(reshape(skeletonData(k,dim,1:(end-1)),[],1), reshape(skeletonData(k,dim, 2:end),[],1));
                            temp = temp + abs(C(1,2));
                        end
                        temporalCov(k) = temporalCov(k)+temp;
                    end

                    count = count + 1;

                end
            end
        end
    end

    temp = spatialCov(~isnan(spatialCov));
    spatialCorrM = mean(temp);
    temp = temporalCov(~isnan(temporalCov));
    temporalCorrM = mean(temp);
    TtoS_ratio = temporalCorrM / spatialCorrM 
    % 1.1358
    
elseif (strcmp(dataset, 'Florence3D'))    
    
    load('../LieGroup_combine_GFT/data/Florence3D/skeletal_data');
    load('../LieGroup_combine_GFT/data/Florence3D/body_model');
    
    Es = [ 2 2 2 2 4 5 7 8 3 10 11 3 13 14;
          1 3 4 7 5 6 8 9 10 11 12 13 14 15];
% Es = [3 3 3 3  1 8  2 9  4 7 7 5  14 6  15;
%       1 2 4 20 8 10 9 11 7 5 6 14 16 15 17]; % disconnect the 4 limb-end joints   
   
    Et = [1:param.numTempNode-1 ; 2:param.numTempNode];
    
    n_sequences = length(skeletal_data);
    n_joints = 15;

    spatialCov = zeros(size(Es,2),1);
    temporalCov = zeros(n_joints,1);
    
    for count = 1:n_sequences      

        joint_locations = skeletal_data{count}.joint_locations;
        S = size(joint_locations);
        % Wrap to match the graph index (resize the skeletonData as a matrix of Nx3xT)     
        skeletonData = zeros(S(2),S(1),S(3));
        for t = 1:S(3)
            skeletonData(:,:,t) = joint_locations(:,:,t)';
        end
        % calculate pairwise spatial cov
        for k = 1:size(Es,2)
            temp = 0;
            for dim = 1:3
                C = corrcoef(reshape(skeletonData(Es(1,k),dim,:),[],1), reshape(skeletonData(Es(2,k),dim,:),[],1));
                temp = temp + abs(C(1,2));
            end
            spatialCov(k) = spatialCov(k)+temp;
        end
        % calculate temporal cov
        for k = 1:n_joints
            temp = 0;
            for dim = 1:3
                C = corrcoef(reshape(skeletonData(k,dim,1:(end-1)),[],1), reshape(skeletonData(k,dim, 2:end),[],1));
                temp = temp + abs(C(1,2));
            end
            temporalCov(k) = temporalCov(k)+temp;
        end     
    end
    temp = spatialCov(~isnan(spatialCov));
    spatialCorrM = mean(temp);
    temp = temporalCov(~isnan(temporalCov));
    temporalCorrM = mean(temp);
    TtoS_ratio = temporalCorrM / spatialCorrM 
    % 
end

end
