function accuracy = classificationError(X, y, nameOfClassifier)
%CLASSIFICATIONERROR finds the accuracy of predicted data (X) against
%true values (y)

    %find the accuracy 
    accuracy = mean(X == y) * 100;
    
    %find the error
    error = 100-accuracy;
    
    fprintf('Classification Error of %s: %f%%\n', nameOfClassifier, error);
    
end

