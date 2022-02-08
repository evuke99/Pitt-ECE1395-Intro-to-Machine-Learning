clear;
clc;

%% Part 1: Bagging and Handwritten-digits classification

%% Part 1.a: Load the .mat file and pick data

load('./input/HW8_data1.mat');

%get random 25 images
indicies = randperm(length(X)); %gives random permutation of all row indicies in feature matrix

rng25images = X(indicies(1:25),:);

%Display 25 random images
figure;
for i = 1:size(rng25images, 1)  
    
   subplot(5, 5, i); 
   image = reshape(rng25images(i,:), [20 20]);
   imshow(image);
   
end

%% Part 1.b: randomly split the data into train and test sets
   
[X_train, y_train, X_test, y_test] = getTrainAndTestMatrix(X, y, 0.9);

%% Part 1.c: Apply bagging on the training set

%equaly bag the training set
[X1, Y1, X2, Y2, X3, Y3, X4, Y4, X5, Y5] = equalBags(X_train, y_train);

%save all matricies to the input file
save('./input/X1.mat', 'X1');
save('./input/Y1.mat', 'Y1');
save('./input/X2.mat', 'X2');
save('./input/Y2.mat', 'Y2');
save('./input/X3.mat', 'X3');
save('./input/Y3.mat', 'Y3');
save('./input/X4.mat', 'X4');
save('./input/Y4.mat', 'Y4');
save('./input/X5.mat', 'X5');
save('./input/Y5.mat', 'Y5');

%% Part 1.d: Train One-vs-One SVM using X1

%train the model based on X1 and Y1
[trainedSVMClassifier_X1, validationError_X1] = trainSVMClassifier(X1, Y1);

%Compute the classification error on the training set 𝑋1.
prediction = trainedSVMClassifier_X1.predictFcn(X1);
classifierAccuracy_X1_SVM = classificationError(prediction, Y1, 'X1 trained SVM Classifier on X1 Dataset');

%Compute the classification error on the testing set. 
prediction_SVM1 = trainedSVMClassifier_X1.predictFcn(X_test);
classifierAccuracy_X1_SVM_test = classificationError(prediction_SVM1, y_test, 'X1 trained SVM Classifier on test Dataset');

%% Part 1.e: Train One-vs-One SVM using X2 

%train the model based on X2 and Y2
[trainedSVMClassifier_X2, validationError_X2] = trainSVMClassifier(X2, Y2);

%Compute the classification error on the training set 𝑋2.
prediction = trainedSVMClassifier_X2.predictFcn(X2);
classifierAccuracy_X2_SVM = classificationError(prediction, Y2, 'X2 trained SVM Classifier on X2 Dataset');

%Compute the classification error on the testing set. 
prediction_SVM2 = trainedSVMClassifier_X2.predictFcn(X_test);
classifierAccuracy_X2_SVM_test = classificationError(prediction_SVM2, y_test, 'X2 trained SVM Classifier on test Dataset');

%% Part 1.f: Train a KNN classifier where K = 7 and training data is X3

%train the model based on X3 and Y3
[trainedKNNClassifier_X3, validationError_X3] = trainSVMClassifier(X3, Y3);

%Compute the classification error on the training set 𝑋3.
prediction = trainedKNNClassifier_X3.predictFcn(X3);
classifierAccuracy_X3_KNN = classificationError(prediction, Y3, 'X3 trained KNN Classifier on X3 Dataset');

%Compute the classification error on the testing set. 
prediction_KNN = trainedKNNClassifier_X3.predictFcn(X_test);
classifierAccuracy_X3_KNN_test = classificationError(prediction_KNN, y_test, 'X3 trained KNN Classifier on test Dataset');

%% Part 1.g: Train a decision tree classifier using training data X4

%train the model based on X4 and Y4
[trainedTreeClassifier_X4, validationError_X4] = trainTreeClassifier(X4, Y4);

%Compute the classification error on the training set 𝑋4.
prediction = trainedTreeClassifier_X4.predictFcn(X4);
classifierAccuracy_X4_Tree = classificationError(prediction, Y4, 'X4 trained Tree Classifier on X4 Dataset');

%Compute the classification error on the testing set. 
prediction_Tree = trainedTreeClassifier_X4.predictFcn(X_test);
classifierAccuracy_X4_Tree_test = classificationError(prediction_Tree, y_test, 'X4 trained Tree Classifier on test Dataset');

%% Part 1.h: Train a random forest classifier using training data X5

%train the model based on X5 and Y5
[trainedRandomForestClassifier_X5, validationError_X5] = trainRandomForestClassifier(X5, Y5);

%Compute the classification error on the training set 𝑋5.
prediction = trainedRandomForestClassifier_X5.predictFcn(X5);
classifierAccuracy_X5_RandomForest = classificationError(prediction, Y5, 'X5 trained Random Forest Classifier on X5 Dataset');

%Compute the classification error on the testing set. 
prediction_RF = trainedRandomForestClassifier_X5.predictFcn(X_test);
classifierAccuracy_X5_RandomForest_test = classificationError(prediction_RF, y_test, 'X5 trained Random Forest Classifier on test Dataset');

%% Part 1.i: Use Majority Voting to combine outputs and report error rate

%initialize majority vote array
majorityVote = zeros(size(X_test,1), 1);

%loop over all examples in X_test
for i = 1:size(X_test, 1)
    
    %store all predictions in prediction vector
    prediction = [prediction_SVM1(i) prediction_SVM2(i) prediction_KNN(i) prediction_Tree(i) prediction_RF(i)];
    
    %find the majority vote
    vote = mode(prediction);
    
    %store vote in majority vote vector
    majorityVote(i) = vote;
    
end

%calculate the classifcation error of the majority vote
majorityVoteError = classificationError(majorityVote, y_test, 'Majority Vote');

%% Part 2: K-means clustering and image segmentation

%intialize clusters(K), iterations, and repetitions(R)
K = [2 3 5 7];
iters = [7 13 20];
R = [5 15 25];

%% Part 2.a: Implement kmeans_single to find the cluster means and associated IDs

X = rand(10, 3); %test value 

[ids, means, ssd] = kmeans_single(X, 2, 6);

%% Part 2.b: Implement multiple runnings of kmeans_single

[ids, means, ssd] = kmeans_multiple(X, 7, 7, 5);

%% Part 2.c: Apply clustering to segment and recolor four different images

%load the images
im1 = imread('./input/HW8_images/im1.jpg');
im2 = imread('./input/HW8_images/im2.jpg');
im3 = imread('./input/HW8_images/im3.png');

%counter variable used for file writing
count = 0;

%loop over all clusters(k), iterations, and repetitions(R)
for k = K
    for iterations = iters
        for r = R
            count = count + 1;
            %process the first image
            im1_out = Segment_kmeans(im1, k, iterations, r);
            s1 = sprintf('Image 1 Processed: Clusters(K) = %d, Iterations = %d, Resets(R) = %d', k, iterations, r);
            disp(s1);
            filename = sprintf('./output/im1_%d.jpg', count);
            imwrite(im1_out, filename);
            
            %process the second image
            im2_out = Segment_kmeans(im2, k, iterations, r);
            s2 = sprintf('Image 2 Processed: Clusters(K) = %d, Iterations = %d, Resets(R) = %d', k, iterations, r);
            disp(s2);
            filename = sprintf('./output/im2_%d.jpg', count);
            imwrite(im2_out, filename);
            
            %process the third image
            im3_out = Segment_kmeans(im3, k, iterations, r);
            s3 = sprintf('Image 3 Processed: Clusters(K) = %d, Iterations = %d, Resets(R) = %d', k, iterations, r);
            disp(s3);
            filename = sprintf('./output/im3_%d.png', count);
            imwrite(im3_out, filename);           
           
        end
    end
end


















