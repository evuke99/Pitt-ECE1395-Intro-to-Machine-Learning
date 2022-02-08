function [y_predict, distance, magnitude] = weightedKNN(X_train, y_train, X_test, sigma)

    n = length(y_train);

    distance = pdist2(X_test, X_train);
    
    magnitude = -(distance).^2;
    
    w = exp(magnitude/(sigma^2));
    
    y_predict = zeros(25,1);
    
    for i = 1:25

        weightedSum = [0 0 0];

        for j = 1:125

            if(y_train(i,1) == 1)
                weightedSum(1) = weightedSum(1) + w(i, j);
            elseif (y_train(i,1) == 2)
                weightedSum(2) = weightedSum(2) + w(i, j);
            elseif (y_train(i,1) == 3)
                weightedSum(3) = weightedSum(3) + w(i, j);

            end
        end

        if (weightedSum(1) > weightedSum(2)) && (weightedSum(1) > weightedSum(3))
            y_predict(i,1) = 1;
        elseif (weightedSum(2) > weightedSum(1)) && (weightedSum(2) > weightedSum(3))
            y_predict(i,1) = 2;
        elseif (weightedSum(3) > weightedSum(2)) && (weightedSum(3) > weightedSum(1))
            y_predict(i,1) = 3;
            
        end
    end
    
end