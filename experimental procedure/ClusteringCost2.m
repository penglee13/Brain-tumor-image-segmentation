function [z] = ClusteringCost2(m, X)

    % Calculate Distance Matrix
    g=reshape(m,3,4)'; % create a cluster center matrix(4(clusters) points in 3(features) dim plane)=[4x3]
    d = pdist2(X, g); % create a distance matrix of each data points in input to each centers = [(s1*s2)x4]

    % Assign Clusters and Find Closest Distances
    [dmin, ind] = min(d, [], 2);
    % ind value gives the cluster number assigned for the input = [(s1*s2)x1]
    
    % Sum of Within-Cluster Distance
    WCD = sum(dmin); 
    
    z=WCD; % fitness function contain sum of each data point to their corresponding center value set (aim to get it minimum)    
    % z = [1 x 1]     
end