function [ids, means, ssd] = kmeans_multiple(X, K, iters, R)

    %initalize arrays/cell arrays to store all ssd/means/ids
    ssdAll = zeros(R,1);
    meansAll = {R};
    idsAll = {R};
    
    %loop over all restarts
    for restarts = 1:R
        %store all ids, means, and ssds
        [idsAll{restarts}, meansAll{restarts}, ssdAll(restarts)] = kmeans_single(X, K, iters);
    end
    
    %find smallest ssd and its array index
    [ssd, index] = min(ssdAll);
    
    %return means/ids arrays based on smallest ssd index
    means = meansAll{index};
    ids = idsAll{index};

end

