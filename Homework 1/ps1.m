clear;
clc;

% diary ps1_report

%% 3.a) Generate 1000000x1 vector of gaussian (normal) distributions
% mean = 1 and standard deviation = 2.8

m = 1;
std_dev = 2.8;
x = m + std_dev.*randn(1e6, 1);

%% 3.b) What is the min/max of x, what is the mean, what is the std deviation

min_x = min(x)
max_x = max(x)
mean_x = mean(x)
stdD_x = std(x)

%% 3.c) plot normalized histogram of x

hist(x); %hist is not reccomended to be used, but i provided both in report
histogram(x);

%% 3.d) Use loop to add 1 to every value in x and time the operation

loop_x = x;
% all ways to find size of column vector
% Size_x = length(x);
% Size_x = size(x, 2);
% Size_x = max(size(x));

tic;

for i = 1:1:length(x)
    loop_x(i,:) = x(i,:) + 1;
end

toc;

%% 3.e) Iterate though the original x and then add 1 without using a loop

noLoop_x = x;

tic;

noLoop_x = x + 1;

toc;

%% 3.f) define vector y that contains all values 5 < x < 20

y = [];

for i = 1:1:length(x)
    if (x(i,:) > 5.0) && (x(i,:) < 20.0)
        y(end+1) = x(i,:);
    end
end

%% 4.a) Define matrix A, find min of each col, min in row, smallest value in A, then create matrix B whos elements are the square of A

A = [2 1 3;...
     2 6 8;...
     6 8 18];
 
 minCol_A = min(A, [], 1)
 minRow_A = min(A, [], 2)
 
 min_A = min(A, [], 'all')
 
 B = sqrt(A)
 
 %% 4.b) solve system of linear equations
 
equ = [2 1 3;...
       2 6 8;...
       6 8 18];
   
answers = [1;...
           3;...
           5];
       
xyz = linsolve(equ, answers)

%% 4.c) Compute the norms
x1 = [0.5 0 -1.5];
x2 = [1 -1 0];

L1_x1 = norm(x1, 1)
L2_x1 = norm(x1, 2)

L1_x2 = norm(x2, 1)
L2_x2 = norm(x2, 2)
 

%% 5) Use the created normalize_col function

input1 = randn(randi(10), randi(10))
output1 = normalize_col(input1)

test_input1 = sum(output1)

input2 = randn(randi(10), randi(10))
output2 = normalize_col(input2)

test_input2 = sum(output2)

% diary off










