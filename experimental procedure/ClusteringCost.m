function [z] = ClusteringCost(m, X)

    % Calculate Distance Matrix
    g=reshape(m,2,3)'; % create a cluster center matrix(3(clusters) points in 2(features) dim plane)=[3x2]
    d = pdist2(X, g); % create a distance matrix of each data points in input to each centers = [15x3]

    % Assign Clusters and Find Closest Distances
    [dmin, ind] = min(d, [], 2);
    % ind value gives the cluster number assigned for the input = [15x1]
    
    % Sum of Within-Cluster Distance
    WCD = sum(dmin); 
    
    z=WCD; % fitness function contain sum of each data point to their corresponding center value set (aim to get it minimum)    
    % z = [1(inputs combinations) x 1]     
end