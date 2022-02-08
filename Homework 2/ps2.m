clc
clear

testData = [1 2;...
            2 4;...
            3 6;...
            4 8;];
%%        
X = testData(:,1); %extract column 1
y = testData(:,2); %extract column 2

X = [ones(length(X), 1), X]; %[feature_0 feature_1]
        
theta_i = [0 0.5]; %transpose the matrix for multiplication

J_i = computeCost(X, y, theta_i);

theta_ii = [3.5 0];
J_ii = computeCost(X, y, theta_ii);

%%
alpha = 0.001;
iter = 15;

[theta_GD, cost] = gradientDescent(X, y, alpha, iter);

theta_normal = normalEqn(X(:,2), y);

%%
filename = 'hw2_data1.txt';

dataIn = importdata(filename);

population = dataIn(:,1); %X
profit = dataIn(:, 2);    %Y



popIntercept = [ones(length(population), 1) population]; %add column of 1's as intercept theta_naught

size(popIntercept);
size(profit);

X_train = [popIntercept(1:round(length(popIntercept)*0.9), 1) popIntercept(1:round(length(popIntercept)*0.9), 2)];
y_train = profit(1:round(length(profit)*0.9), 1);
X_test = [popIntercept(round(length(popIntercept)*0.9):end, 1) popIntercept(round(length(popIntercept)*0.9):end, 2)];
y_test = profit(round(length(profit)*0.9):end, 1);

alpha = 0.01;
iterations = 750;

[theta_Q4 cost_Q4]= gradientDescent(X_train, y_train, alpha, iterations);

% hold on;
% plot(cost, iterations);
% xlabel("Iterations")
% ylabel("Cost")

figure
scatter(population, profit, 'x', 'red');
xlabel("Population of City in 10,000s")
ylabel("Profit in $10,000s")

hold on;
plot(X_train, theta_Q4(2)*X_train + theta_Q4(1), 'blue')

iterations = 1:750;
figure
plot(cost_Q4, iterations);
xlabel("Cost")
ylabel("Iterations")

modelParameters = theta_Q4;

y_pred = X_test(:, 1).*modelParameters(1) + X_test(:,2).*modelParameters(2);

predictionCost = computeCost(X_test, y_pred, modelParameters)
testCost = computeCost(X_test, y_test, modelParameters)

theta_Q4g_norm = normalEqn(X_test(:,2), y_train);
predictionCost = computeCost(X_test, y_pred, theta_Q4g_norm);
testCost = computeCost(X_test, y_test, theta_Q4g_norm);

a = [0.0001 0.001 0.003 0.03];

for i = 1:4
    
    alpha = a(i);
    
    [theta, cost]= gradientDescent(X_train, y_train, alpha, 250);
    
    iterations = 1:250;
    figure
    plot(cost, iterations);
    xlabel("Cost")
    ylabel("Iterations")
    legend("alpha = " + a(i))
    
end


%%
clc
clear
filename = 'hw2_data2.txt';

dataIn = importdata(filename);

loadedHouseSize = dataIn(:,1);
loadedBedrooms = dataIn(:,2);
price = dataIn(:,3);

meanHouseSize = mean(loadedHouseSize);
meanBedrooms = mean(loadedBedrooms);

stdHouseSize = std(loadedHouseSize);
stdBedrooms = std(loadedBedrooms);

houseSize = (loadedHouseSize - meanHouseSize)./stdHouseSize;
bedrooms = (loadedBedrooms - meanBedrooms)./stdBedrooms;

features = [ones(length(houseSize),1) houseSize bedrooms];
sizeFeatures = size(features);

alpha = 0.01;
iterations = 750;
[theta_Q5, cost_Q5] = gradientDescentMultiVariable(features, price, alpha, iterations);

iterations = 1:750;
figure
plot(cost_Q5, iterations);
xlabel("Cost")
ylabel("Iterations")

x0 = 1;
x1 = (1570 - meanHouseSize)./stdHouseSize;
x2 = (4 - meanBedrooms)./stdBedrooms;

predictionVector = [x0 x1 x2];

[theta_Q5_c, cost_Q5_c] = gradientDescentMultiVariable(predictionVector, price, alpha, iterations);

prediction = theta_Q5_c(1)*predictionVector(1) + theta_Q5_c(2)*predictionVector(2) + theta_Q5_c(3)*predictionVector(3)










