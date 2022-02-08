function J = computeCost(X, y, theta)

    %FUNCTION PURPOSE: Computes the cost/error of a feature/label vector
    %based on calculated theta

    m = length(y); %number of training samples 

    J = 0;
    
    J = (1/(2*m))*sum((X*theta - y).^2);

end