function [J, grad] = costFunction(theta, X_train, y_train)

    m = size(X_train); %gets the amount of training examples
    hx = sigmoid(X_train*theta); %logistic hypothesis
    
    J = 0;
    grad = 0;
    
    %cost function portion
    
    summation = -y_train.*log(hx) - (1-y_train).*log(1-hx);
    
    J = (1/m(1))*sum(summation); %can use sum becuase summation is a vector
        
    %gradient descent portion
    
    for i = 1:m(1) %need to use for loop here becuase X_train is a matrix
        
        grad = grad + (hx(i) - y_train(i))*X_train(i,:);
    
    end
    
    grad = (1/m(1))*grad;
        
end