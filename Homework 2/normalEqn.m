function theta = normalEqn(X_train, y_train)

    one = pinv(X_train'*X_train);
    two = one*X_train'
    three = two.*y_train;


end