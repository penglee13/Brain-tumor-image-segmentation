clc;
clear;
close all;

%% Problem Definition

X = [1 1.5;2 2.5;3 3.5;1 1.5;2 2.5;3 3.5;1 1.5;2 2.5;3 3.5;1 1.5;2 2.5;3 3.5;1 1.5;2 2.5;3 3.5]; % [15x2]
k = 3; % no. of clusters

CostFunction=@(m) ClusteringCost(m, X);     % Cost Function m = [3x2] cluster centers

VarSize=[k size(X,2)];  % Decision Variables Matrix Size = [3 2]

nVar=prod(VarSize);     % Number of Decision Variables = 6

VarMin= repmat(min(X),1,k);      % Lower Bound of Variables [3x1] of[1x2] = [3x2]
VarMax= repmat(max(X),1,k);      % Upper Bound of Variables [3x1] of[1x2] = [3x2]

ga_opts = optimoptions('particleswarm','display','iter','MaxTime',600);
[centers, err_ga] = particleswarm(CostFunction, nVar,VarMin,VarMax,ga_opts);

m=centers;

    g=reshape(m,2,3)'; % create a cluster center matrix(3(clusters) points in 2(features) dim plane)=[3x2]
    d = pdist2(X, g); % create a distance matrix of each data points in input to each centers = [15x3]

    % Assign Clusters and Find Closest Distances
    [dmin, ind] = min(d, [], 2)
    % ind value gives the cluster number assigned for the input = [15x1]
    
    % Sum of Within-Cluster Distance
    WCD = sum(dmin); 
    
    z=WCD; % fitness function contain sum of each data point to their corresponding center value set (aim to get it minimum)    
    % z = [1(inputs combinations) x 1]     

