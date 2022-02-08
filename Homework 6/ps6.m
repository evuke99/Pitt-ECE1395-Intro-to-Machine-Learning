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

%% Question 1

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

sizeOfX_train_0 = size(X_train_0) %output
sizeOfX_train_1 = size(X_train_1) %output

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

mean_std_table = table(feature, class, mean, std) %output

%% Part c
 P_w0 = 0.65;
 P_w1 = 0.35;
 
 results = zeros(1,228);
 
 numCorrect = 0;

 for i = 1:228
     
      P_Xj_w0 = zeros(1,8);
      P_Xj_w1 = zeros(1,8);
      
      P_X_w0 = 1;
      P_X_w1 = 1;
      
     for j = 1:8

        P_Xj_w0(j) = sum(1/(sqrt(2*pi) * stdClass0(j)) * exp(-(X_test(i,j) - meanClass0(j)).^2/(2*stdClass0(j))));
        P_Xj_w1(j) = sum(1/(sqrt(2*pi) * stdClass1(j)) * exp(-(X_test(i,j) - meanClass1(j)).^2/(2*stdClass1(j))));
        
     end
     
     for j = 1:8

        P_X_w0 = P_X_w0 .* P_Xj_w0(1,j);
        P_X_w1 = P_X_w1 .* P_Xj_w1(1,j);
    
     end
     
     P_w0_X = P_X_w0 * P_w0;
     P_w1_X = P_X_w1 * P_w1;
        
     if(P_w0_X >= P_w1_X)
         results(i) = 0;
     else
         results(i) = 1;
     end
     
     if(results(i) == y_test(i))
         numCorrect = numCorrect + 1;
     end

 end

accuracy = numCorrect/length(y_test) %output

%% Question 2

%% Part a

C = cov(X_train); %output screenshot of matrix
sizeC = size(C) %output

%% Part b

%mean vectors calculated above

%% Part c

sizeXTest = size(X_test);
results = zeros(sizeXTest(1), 1);
numCorrect = 0;

for i = 1:sizeXTest(1)
    
    %had to switch transpose becuase of meanClass being a row vector
    d0 = sqrt((X_test(i,:) - meanClass0')' * inv(C) *  (X_test(i,:) - meanClass0'));
    d1 = sqrt((X_test(i,:) - meanClass1')' * inv(C) *  (X_test(i,:) - meanClass1'));
    
    %account for the non-equal probabilites
    d0 = d0 - 2*log(P_w0);
    d1 = d1 - 2*log(P_w1);
    
    if d0 >= d1
        results(i) = 0;
    else
        results(i) = 1;
    end
    
    if(results(i) == y_test(i))
        numCorrect = numCorrect + 1;
    end
end

accuracy = numCorrect/length(y_test) %output


     
     
     
 
 
     
 

