function [ids, means, ssd] = kmeans_single(X, k, iters)

    %number of features
    n = size(X, 2);
    
    %number of samples
    samples = size(X,1);

    %matrix containting centers/means for each cluster
    means = zeros(k, n);
    
    %randomly initialze the means
    
    %take max and min of feature dimention
    maxFeatures = max(X);
    minFeatures = min(X);
    
    %find the range of each of the features
    rangeFeatures = abs(maxFeatures - minFeatures);
    
    %initialize the random variable
    random = zeros(1,n);

    %randomly initialize the means
    for cluster = 1:k
        
        for j = 1:n
            random(j) = rangeFeatures(j)*rand(1) + minFeatures(j);
        end
        
        means(cluster,:) = random;
    
    end
        
    %loop over iterations
    for i = 1:iters
        
        %calculate the distance between image and means
        distance = pdist2(X,means);
        
        %initialize the IDs matrix
        ids = zeros(samples,1);

        %find index of minimun distance value
        [~, ids] = min(distance, [], 2);
    
        %loop over all clusters
        for cluster = 1:k
            
            %initialize variables
            sum = zeros(1,n);
            count = 0;

            %loop over all samples/examples
            for j = 1:samples

                %check if the ID of the current sample matches the current
                %cluster
                if(ids(j) == cluster)
                    sum = sum + X(j,:); %if it does, sum those pixels
                    count = count + 1;
                end
            end

            %find avg of each column in sum
            means(cluster,:) = sum/count;

        end
        
    end
    
    %compute the distance again based on new means
    distance = pdist2(X,means);
    temp = 0;
    
    %loop over all samples
    for index_ssd = 1:samples
        %calculate ssd in temp variable and add distance squared
        temp = temp + distance(index_ssd, ids(index_ssd))^2;
    end
    
    %output distance
    ssd = temp;

end

