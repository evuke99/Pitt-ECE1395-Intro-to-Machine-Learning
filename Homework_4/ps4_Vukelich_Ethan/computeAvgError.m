function [avgTrainingError, avgTestingError] = computeAvgError(X_data, y, lambda)

    %FUNCTION PURPOSE: Computes the average error of a Feature Vector and
    %Label Vector based on lambda values using a regularized normal
    %equation for linear regression and then displays it on a lambda vs
    %average error plot

    for i = 1:20
    
        [X_train, y_train, X_test, y_test] = getTrainAndTestMatrix(X_data, y, 0.88);

        for j = 1:length(lambda)

            theta = Reg_normalEqn(X_train,y_train,lambda(j));

            trainingError(i,j) = computeCost(X_train, y_train, theta);

            testingError(i,j) = computeCost(X_test, y_test, theta);

        end
    end

    avgTrainingError = mean(trainingError);
    avgTestingError = mean(testingError);


end