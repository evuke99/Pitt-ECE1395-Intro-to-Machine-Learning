clear
clc

%% 1.a) Make the Reg_normalEqn.m function

%% 1.b) Load dataset into matlab

filename = './input/hw4_data1.mat';

load(filename);

X_data = addX0(X_data);  %add X0 = 1

sizeFeatureMatrix = size(X_data)

%% 1.c) Compute average training and testing sets

lambda = [0 0.001 0.003 0.005 0.007 0.009 0.012 0.017];

[avgTrainingError, avgTestingError] = computeAvgError(X_data, y, lambda); %function developed by me

figure
plot(lambda, avgTrainingError, 'r-o', lambda, avgTestingError, 'b-o')
xlabel('lambda')
ylabel('Average Error')
legend('Training Error', 'Testing Error')

saveas(gcf, './output/ps4-1-a.png')

%% 2) KNN - Effect of K

clear
clc

filename = './input/hw4_data2.mat';

load(filename);

% First Classifier Data
X_train1 = [X1;X2;X3;X4];
y_train1 = [y1;y2;y3;y4];
X_test1 = X5;
y_test1 = y5;

% Second Classifier Data
X_train2 = [X1;X2;X3;X5];
y_train2 = [y1;y2;y3;y5];
X_test2 = X4;
y_test2 = y4;

% Third Classifier Data
X_train3 = [X1;X2;X4;X5];
y_train3 = [y1;y2;y4;y5];
X_test3 = X3;
y_test3 = y3;

% Fourth Classifier Data
X_train4 = [X1;X3;X4;X5];
y_train4 = [y1;y3;y4;y5];
X_test4 = X2;
y_test4 = y2;

% Fifth Classifier Data
X_train5 = [X2;X3;X4;X5];
y_train5 = [y2;y3;y4;y5];
X_test5 = X1;
y_test5 = y1;

avgAccuracy = [];

for k = 1:2:15
    
    modelformed = fitcknn(X_train1, y_train1, 'NumNeighbors', k, 'Standardize', 1);
    predict1 = predict(modelformed, X_test1);
    
    correct1 = nnz(predict1 == y1);
    
    modelformed = fitcknn(X_train2, y_train2, 'NumNeighbors', k, 'Standardize', 1);
    predict2 = predict(modelformed, X_test2);
    
    correct2 = nnz(predict2 == y2);
    
    modelformed = fitcknn(X_train3, y_train3, 'NumNeighbors', k, 'Standardize', 1);
    predict3 = predict(modelformed, X_test3);
    
    correct3 = nnz(predict3 == y3);
    
    modelformed = fitcknn(X_train4, y_train4, 'NumNeighbors', k, 'Standardize', 1);
    predict4 = predict(modelformed, X_test4);
    
    correct4 = nnz(predict4 == y4);
    
    modelformed = fitcknn(X_train5, y_train5, 'NumNeighbors', k, 'Standardize', 1);
    predict5 = predict(modelformed, X_test5);
    
    correct5 = nnz(predict5 == y5);
    
    avgAccuracy(length(avgAccuracy)+1) = (correct1 + correct2 + correct3 + correct4 + correct5)/5;

end

accuracyPlot = [(1:2:15)' avgAccuracy'];

figure
plot(accuracyPlot(:,1), accuracyPlot(:,2));
xlabel('K')
ylabel('Accuracy')

saveas(gcf, './output/ps4-2-a.png')


%% 2) Weighted KNN

clear
clc

filename = './input/hw4_data3-1.mat';

load(filename);

sigma = [0.01 0.1 0.5 1 3 5];

accuracy = [0 0 0 0 0 0];

t = 0;
for i = sigma
    
    [y_predict, distance, magnitude] = weightedKNN(X_train, y_train, X_test, i);
    
    for j = 1:25
        if y_predict(j) == y_test(j)
            accuracy(t+1) = accuracy(t+1) + 1;
        end
    end
    t = t+1;
    
end

figure
plot(sigma, accuracy)
xlabel('sigma')
ylabel('accuracy')
saveas(gcf, './output/ps4-3-b.png')







        
        
        
        
