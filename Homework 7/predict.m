function P = predict(Theta1,Theta2, X)

    m = size(X,1);

    % compute z and a for each layer
    a1 = [ones(m, 1) X];
    z1 = a1* Theta1';
    a2 = sigmoid(z1);

    a2 = [ones(m, 1) a2];
    z2 = a2 * Theta2';
    a3 = sigmoid(z2);

    % P returns the index of the column that has the max value of h3
    [~, P] = max(a3, [], 2);


end

