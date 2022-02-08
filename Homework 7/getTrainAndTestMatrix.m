function [X_train, y_train, X_test, y_test] = getTrainAndTestMatrix(X, y, trainPercent)

    %FUNCTION PURPOSE: Splits feature vector and label vector into training
    %and testing sets
    
    %trainPercent must be a decimal between 0 and 1

    indicies = randperm(length(X)); %gives random permutation of all row indicies in feature matrix
    
    X_train = X(indicies(1:round(length(indicies)*trainPercent)), :);
    y_train = y(indicies(1:round(length(indicies)*trainPercent)), :);

    X_test = X(indicies((round(length(indicies)*trainPercent))+1:end),:);
    y_test = y(indicies((round(length(indicies)*trainPercent))+1:end));
    
end