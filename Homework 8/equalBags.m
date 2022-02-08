function varargout = equalBags(X, y)
%EQUALBAGS: applies bagging to a set of data and returns a number of bags
%and their corresponding lables specified by the output arguments in the 
%function call

%input X: feature matrix with rows = examples and cols = features
%input y: label vector, must be column vector and match size of X

%output arguments must be in form [X1, Y1, X2, Y2, ...] 
%number of bags determend by [X,Y] pairs

%bags will always be of equal size

%if number of examples does not evently divide into amount of bags
%requested, some examples will be excluded to compinsate
%the excluded examples will be truncated off the end of the feature matrix

    %sets up varargout
    varargout = cell(1,nargout);
    
    %determine number of bags by [X,Y] pairs
    numBags = nargout/2;
    
    %calculate even bag sizes, round, set truncated amount if necessary
    bagSize = round(size(X, 1)/numBags);
    
    j = 1;
    
    %randomly initialize indicies based on number of examples
    indicies = randperm(size(X,1));
    
    %loop over number of bags
    for k = 1:numBags
        
        %evenly distribute bag/bag_labels contents based on random indicies
        bag = X(indicies((bagSize*k)-(bagSize-1):(bagSize*k)), :);
        bag_labels = y(indicies((bagSize*k)-(bagSize-1):(bagSize*k)), :);
        
        %populate cell array
        varargout{j} = bag;
        varargout{j+1} = bag_labels;
        
        j = j + 2;
        
    end
    

end

