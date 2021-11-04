clear;
clc;

%%load data into matrix
dataIn = table2array(readtable('./input/diabetes.csv'));

%add bias feature X0
% dataIn = [ones(length(dataIn), 1) dataIn];  

%get size of dataIn
sizeDataIn = size(dataIn);

%get training and testing data
[X_train, y_train, X_test, y_test] = getTrainAndTestMatrix(dataIn(:,1:sizeDataIn(2) - 1), dataIn(:,sizeDataIn(2)), 0.7031);

%% Part 1

%% Part a

%%seperate training matrix into two training matricies based on class
X_train_0 = [];
X_train_1 = [];
count0 = 0;
count1 = 0;

%%if unknown errors in the end, check here first
for i = 1:length(y_train)
    if(y_train(i) == 0)
        X_train_0(count0+1, :) = X_train(i, :);
        count0 = count0 + 1;
    else
        X_train_1(count1+1, :) = X_train(i, :);
        count1 = count1 + 1;
    end
end

sizeOfX_train_0 = size(X_train_0)
sizeOfX_train_1 = size(X_train_1)

%% Part b

%Calculate the mean and standard deviation of each feature for each class
meanClass0 = mean(X_train_0);
meanClass1 = mean(X_train_1);
stdClass0 = std(X_train_0);
stdClass1 = std(X_train_0);
    
% make the table
feature = ['x1';'x2';'x3';'x4';'x5';'x6';'x7';'x8';'x1';'x2';'x3';'x4';'x5';'x6';'x7';'x8'];
class = [0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1];
mean = horzcat(meanClass0, meanClass1)';
std = horzcat(stdClass0, stdClass1)';

mean_std_table = table(feature, class, mean, std)

%% Part c
 P_w0 = 0.65;
 P_w1 = 0.35;
 
 P_Xj_w0 = zeros(228,8);
 P_Xj_w1 = zeros(228,8);

 
 %i

 for j = 1:8
     
    coeff0 = 0;
    exponent0 = zeros(228,1);
    coeff1 = 0;
    exponent1 = zeros(228,1);
    
    coeff0 = 1/(sqrt(2*pi)*stdClass0(j));
    exponent0 = exp( ( -(X_test(:,j) - meanClass0(j)) ) / (2 * stdClass0(j)^2));
    P_Xj_w0(:,j) = coeff0 .* exponent0;
    
    coeff1 = 1/(sqrt(2*pi)*stdClass1(j));
    exponent1 = exp( ( -(X_test(:,j) - meanClass1(j)) ) / (2 * stdClass1(j)^2));
    P_Xj_w1(:,j) = coeff1 .* exponent1;

 end

% pd = normpdf(X_test(:,1), meanClass0(1), stdClass0(1));
% 
% plot(X_test(:,1), pd)

 
 %ii
 

     
     
     
 
 
     
 

