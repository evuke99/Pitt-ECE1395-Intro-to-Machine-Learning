function [J, grad] = costFunction(theta, X_train, y_train)

    m = size(X_train); %gets the amount of training examples
    hx = sigmoid(theta*X_train); %logistic hypothesis
    
    grad = 0;
    
    %cost function portion
    
    summation = -y_train.*log(hx) - (1-y_train).*log(1-hx);
    
    J = (1/m(2))*sum(summation);
    
    %gradient descent portion
    
    for i = 1:m(2)
        
        grad = grad + (hx(i) - y_train(i))*X_train(i,:);
    
    end
    
    grad = (1/m)*grad;
        
end

