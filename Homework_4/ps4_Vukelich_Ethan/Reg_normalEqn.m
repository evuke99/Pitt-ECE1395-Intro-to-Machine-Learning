function theta = Reg_normalEqn(X_train,y_train,lambda)

    %FUNCTION PURPOSE: Computes theta value based on the regularized normal
    %equation

    n = size(X_train);

    temp = pinv(X_train'*X_train + lambda.*eye(n(2)));

    theta = temp * (X_train'*y_train);

end