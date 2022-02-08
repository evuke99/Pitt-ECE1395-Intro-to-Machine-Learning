function X_X0 = addX0(X)

    %FUNCTION PURPOSE: Adds a column of 1's to front of matrix

    X_X0 = [ones(length(X), 1) X];  %add X0 = 1

end

