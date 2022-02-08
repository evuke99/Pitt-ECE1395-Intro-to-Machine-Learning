clear;
clc;

%% Part 0 - Load Dataset
load('./input/HW7_Data.mat'); %load feature and label vectors
load('./input/HW7_weights_2.mat'); %load weights

sizeX = size(X); %output
sizeY = size(y); %output

%% Part 1 - Forward Propagation

% 1.a)
P = predict(Theta1, Theta2, X);

%% 1.b)

% Find accuracy
accuracy = mean(P == y) * 100;

fprintf('Accuracy of Part 1: %f%%\n', accuracy);

%% Part 2 - Cost Function

J = zeros(3,1);

for i = [0 1 2]
    J(i+1) = nnCost(Theta1, Theta2, X, y, 3, i);
end

lambda = [0;1;2];

J_table = table(lambda, J) %output

%% Part 3 - Derivation of the active function : sigmoid gradient

%example z: expected output ~[0 0.25 0]'
z = [-10 0 10]';

g_prime = sigmoidGradient(z)


%% Part 4 - Backpropagation for gradient of cost functions and stochastic gradient descent

%setting variables for sGD
input_layer_size = 4;
hidden_layer_size = 8;
num_labels = 3;
alpha = 0.01;

fprintf('The Value of alpha used: %f\n', alpha);

%splitting 
[X_train, y_train, X_test, y_test] = getTrainAndTestMatrix(X, y, 0.85);

trainingAccuracy = zeros(hidden_layer_size,1);
testingAccuracy = zeros(hidden_layer_size,1);

J_train = zeros(hidden_layer_size,1);
J_test = zeros(hidden_layer_size,1);

i = 1;

for MaxEpochs = [50 100]
    for lambda = [0 0.01 0.1 1]
        
        %Cost for training
        [Theta1, Theta2] = sGD(input_layer_size, hidden_layer_size, num_labels,...
                               X_train, y_train, lambda, alpha, MaxEpochs);
                           
        %Prediction and Accuracy of training data
        P = predict(Theta1, Theta2, X_train);
        trainingAccuracy(i) = mean(P == y_train) * 100; 
        
        %Cost of training Data
        J_train(i) = nnCost(Theta1, Theta2, X_train, y_train, num_labels, lambda);
        
        %Prediction and Accuracy of testing data
        P = predict(Theta1, Theta2, X_test);
        testingAccuracy(i) = mean(P == y_test) * 100;
        
        %Cost of training Data
        J_test(i) = nnCost(Theta1, Theta2, X_test, y_test, num_labels, lambda);
        
        i = i + 1;
        
    end
end

%% Part 5 - Testing the Network

%Setting Up table variables
lambda = [0;0.01;0.1;1];
trainingAcc_50Epochs = trainingAccuracy(1:(hidden_layer_size/2));
trainingCost_50Epochs = J_train(1:(hidden_layer_size/2));

testingAcc_50Epochs = testingAccuracy(1:(hidden_layer_size/2));
testingCost_50Epochs = J_test(1:(hidden_layer_size/2));

trainingAcc_100Epochs = trainingAccuracy((hidden_layer_size/2)+1:end);
trainingCost_100Epochs = J_train((hidden_layer_size/2)+1:end);

testingAcc_100Epochs = testingAccuracy((hidden_layer_size/2)+1:end);
testingCost_100Epochs = J_test((hidden_layer_size/2)+1:end);

%Printing table
Accuracy_table = table(lambda, trainingAcc_50Epochs, trainingCost_50Epochs,...
                               testingAcc_50Epochs, testingCost_50Epochs,...
                               trainingAcc_100Epochs, trainingCost_100Epochs,...
                               testingAcc_100Epochs, testingCost_100Epochs) %output
