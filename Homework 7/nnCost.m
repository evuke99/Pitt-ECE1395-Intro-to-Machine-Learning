function J = nnCost(Theta1,Theta2, X, y, K, lambda)

    %training examples
    m = size(X, 1);
    T1_size = size(Theta1);
    T2_size = size(Theta2);
    
    % compute z and h(a) for each layer
    a1 = [ones(m, 1) X];
    z1 = a1* Theta1';
    a2 = sigmoid(z1);

    a2 = [ones(m, 1) a2];
    z2 = a2 * Theta2';
    a3 = sigmoid(z2);

   
    % recode y in terms of 0 and 1
    temp = eye(K);
    recodeY = temp(y,:);
    
    % Accuracy part of J
    
    accuracy = recodeY.*log(a3) + (1-recodeY).*log(1-a3);
    
    cost = (-1/m) .* sum(sum(accuracy,2));
    
    % Regularization part of J
    temp1 = 0;
    temp2 = 0;
    
    for i = 1:T1_size(1)
        for j = 1:T1_size(2)
            temp1 = temp1 + Theta1(i,j).^2;
        end
    end
 
    for i = 1:T2_size(1)
        for j = 1:T2_size(2)
            temp2 = temp2 + Theta2(i,j).^2;
        end
    end
    
    reg = (lambda/(2*m)) .* (temp1 + temp2);
        
    J = cost + reg;

end

