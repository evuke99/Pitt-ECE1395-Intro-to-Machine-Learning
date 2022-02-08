function [Theta1, Theta2] = sGD(input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda, alpha, MaxEpochs)

    %FUNCTION PURPOSE: Determines the stoicastic gradient descent for a
    %3-layer neural network

    m = size(X_train, 1);
    
    %randomly initialize Theta1 and Theta2 on the interval [-0.1 0.1]
    Theta1 = -(-0.1) + (0.1-(-0.1)).*rand(hidden_layer_size,input_layer_size+1);
    Theta2 = -(-0.1) + (0.1-(-0.1)).*rand(num_labels,hidden_layer_size+1);
    
    % recode y in terms of 0 and 1
    temp = eye(num_labels);
    recodeY = temp(y_train,:);
    
%     prevJ = 0;
    
    for epoch = 1:MaxEpochs
    
        % compute z and h(a) for each layer i.e) Run the forward pass
        a1 = [ones(m, 1) X_train];
        z2 = a1* Theta1';
        a2 = sigmoid(z2);
        a2 = [ones(m, 1) a2];
        z3 = a2 * Theta2';
        a3 = sigmoid(z3);

        %Calculate the errors
        dirac3 = a3 - recodeY;
        temp = (dirac3*Theta2);
            
        dirac2 = temp(:,2:end).*sigmoidGradient(z2);
        
        %calculate the gradients
        delta1 = dirac2'*a1;
        delta2 = dirac3'*a2;

        %Set first column of Thetas to 0 so the first column is not
        %regularized
        Theta1 = [zeros(hidden_layer_size, 1) Theta1(:,2:end)];
        Theta2 = [zeros(num_labels, 1) Theta2(:,2:end)];

        %Add regularization to the gradient
        D1 = delta1 + lambda*Theta1;
        D2 = delta2 + lambda*Theta2;
       
        %Recompute Thetas
        Theta1 = Theta1 - alpha*D1;
        Theta2 = Theta2 - alpha*D2;
        
        %Find the cost of the new thetas
        J = nnCost(Theta1, Theta2, X_train, y_train, num_labels, lambda);
        
        %ensure first iteration runs and break if erros is < 1E-4
        if(epoch ~= 1)
            if ( abs(J - prevJ) < 1e-4 )
                break;
            end 
        end
        
        prevJ = J;
       
       
        
    end
        

        
        
    
    
    
    
    
        





end

