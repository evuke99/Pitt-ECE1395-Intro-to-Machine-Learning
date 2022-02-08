function [B] = normalize_col(A)
%NORMALIZE_COL Makes the sum of each column of the output matrix equal 1

%assumes that the input A has all columns add up to a non-zero value

B = A./sum(A); %takes each element and divides it against it's column sum


end