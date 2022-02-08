clear
clc

%% 1.a) load data and seperate into feature matrix and label matrix

filename = './input/hw3_data1.txt';

dataIn = importdata(filename);

featureMatrixX = [ones(length(dataIn), 1) dataIn(:, 1) dataIn(:, 2)]; %create feature matrix

labelVectorY = dataIn(:, 3);

sizeFeature = size(featureMatrixX)
sizeLabel = size(labelVectorY)

%% 1.b) plot training data

figure
gscatter(featureMatrixX(:,2), featureMatrixX(:, 3),labelVectorY, 'rg','o+')
legend('Not Admitted', 'Admitted')
xlabel('Exam 1 Score')
ylabel('Exam 2 Score')
grid on

saveas(gcf, ['./output/ps3-1-b.png'])

%% 1.c) seperate the feature matrix and label vector into test matrix/vectors

indicies = randperm(length(featureMatrixX));

X_train = featureMatrixX(indicies(1:length(indicies)*.90), :);
y_train = labelVectorY(indicies(1:length(indicies)*.90), :);

X_test = featureMatrixX(indicies((length(indicies)*.90)+1:end),:);
y_test = labelVectorY(indicies((length(indicies)*.90)+1:end));

%% 1.d) Create Sigmoid Function

z = [-10:10];
gz = sigmoid(z);

figure
plot(gz, z)
xlabel('gz')
ylabel('z')
grid on

saveas(gcf, ['./output/ps3-1-c.png'])

%% 1.e) Implement the cost function and gradient descent for log. regression

%toy dataset

X_toy = [ 1 0 1; 1 0 3; 1 2 0; 1 2 1 ]; %added X0 = 1
y_toy = [ 0; 1; 0; 1];
t = [ 0 0 0 ]';

[J, grad] = costFunction(t, X_toy, y_toy);

J

%% 1.f) Optimization

%set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%run fminunc to obtain the optimal theta
%this function will return theta and the cost

[theta, cost] = fminunc(@(t)(costFunction(t, X_train, y_train)), t, options)


%% 1.g) Plot the decision boundry
y = (-theta(1)*X_train(:,1) - theta(2)*X_train(:,2))/theta(3);

figure
gscatter(featureMatrixX(:,2), featureMatrixX(:, 3),labelVectorY, 'rg','o+')

hold on

plot(X_train(:,2), y)

legend('Not Admitted', 'Admitted', 'Decision Boundry')
xlabel('Exam 1 Score')
ylabel('Exam 2 Score')
grid on

saveas(gcf, ['./output/ps3-1-f.png'])

%% 1.i) Use your hypotheis to predict values of a test data set

test = [ 1 50 75 ];

z = test*theta;

prob = sigmoid(z)

%% 2 Use linear regression to fit a non-linear function

filename = './input/hw3_data2.mat';

dataIn = importdata(filename);

featureMatrixX = [ones(length(dataIn), 1) dataIn(:, 1) dataIn(:,1).^2 ]; %create feature matrix, population x1000
labelVectorY = dataIn(:, 2); %profit

%% 2.a) Using the normal equation to find theta values
theta = normalEqn(featureMatrixX, labelVectorY)

%% 2.b) plot the data and polynomial fitted curve

yplot = theta(1) + theta(2)*featureMatrixX(:,2) + theta(3)*featureMatrixX(:,3);

figure
scatter(featureMatrixX(:,2), labelVectorY)

hold on

plot(featureMatrixX(:,2), yplot)

legend('training data', 'fitted model')
xlabel('Population (x1000)')
ylabel('Profit')
grid on

saveas(gcf, ['./output/ps3-2-b.png'])







