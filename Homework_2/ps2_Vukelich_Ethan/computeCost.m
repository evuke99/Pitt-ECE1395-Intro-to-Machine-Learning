function J = computeCost(X, y, theta)
%computeCost Compute cost for linear regression

    m = length(y); %number of training samples 

    h_theta = X(:, 1).*theta(1) + X(:,2).*theta(2); %hypothisis h(x(i))

    error = (h_theta - y).^2; %(h(x(i)) - y(i))^2

    J = 1/(2*m) * sum(error);

end