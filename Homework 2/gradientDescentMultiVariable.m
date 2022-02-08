function [theta, cost] = gradientDescentMultiVariable(X_train, y_train, alpha, iters)

    m = length(y_train);
    n = size(X_train);
    theta = randn(1,n(2));
    
    for i = 1:iters

        for k = 1:n(1)

            j0 = (1/m)*sum(X_train(k,1)*theta(1) + theta(2)*X_train(k,2) + theta(3)*X_train(k,3) - y_train(k,1));
            j1 = (1/m)*sum((X_train(k,1)*theta(1) + theta(2)*X_train(k,2) + theta(3)*X_train(k,3) - y_train(k,1))*X_train(k,2));
            j2 = (1/m)*sum((X_train(k,1)*theta(1) + theta(2)*X_train(k,2) + theta(3)*X_train(k,3) - y_train(k,1))*X_train(k,3));
            
        end

        temp_theta0 = theta(1) - alpha*j0;
        temp_theta1 = theta(2) - alpha*j1;
        temp_theta2 = theta(3) - alpha*j2;

        theta(1) = temp_theta0;
        theta(2) = temp_theta1;
        theta(3) = temp_theta2;

        cost(i) = computeCostMultiVariable(X_train, y_train, theta);
        

        
    end


end