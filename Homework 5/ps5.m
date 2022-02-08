clc;
clear;

%% Part 0: Data Preprocessing

dataPreprocess();
image1 = imread('.\input\train\1.pgm');
% figure;
% imshow('.\input\train\1.pgm')
% imwrite(image1, '.\output\ps5-0.png');

T = zeros(10304, 320);

%% Part 1: PCA analysis

%% Part 1.a
for i = 1:320
    
    filename = sprintf('.\\input\\train\\%d.pgm', i);
    T(:,i) = reshape(imread(filename), [], 1);
    
end

% figure;
% imshow(T, []);
% imwrite(T, '.\output\ps5-1-a.png')

%% Part 1.b

m = mean(T,2);
mReshaped = reshape(m, 112, 92);
% figure;
% imshow(mReshaped, []);
% imwrite(mReshaped, '.\output\ps5-1-b.png')

%% Part 1.c
A = T - m;
C = A*A';
% figure;
% imshow(C, []);
% imwrite(C, '.\output\ps5-1-c.png')

%% Part 1.d
eigenValues = eig(A'*A);

eigenvaluesDescending = sort(eigenValues, 'descend');

i = 0;
v = 0;

denominator = 0;
for n = 1:320
    denominator = denominator + eigenvaluesDescending(n);
end
  
while v < 0.95
    
    i = i + 1;
    
    numerator = 0;

    for k = 1:i
        numerator = numerator + eigenvaluesDescending(k);
    end
    
    if i == 1
        v(1) = numerator / denominator;
    else
        v(end+1) = numerator / denominator;
    end

end
        
% figure;
% k_plot = 1:k;
% plot(k_plot, v);
% title('k vs v(k)')
% xlabel('k')
% ylabel('v(k)')
% % saveas(gcf, './output/ps5-1-d.png')

%% Part 1.e

[U,D] = eigs(C,k);
temp = U(:,1:8);
eightFaces = zeros(112,92,8);
for i = 1:8
    eightFaces(:,:,i) = reshape(temp(:,i),[112,92]);
end

% figure;
% for i = 1:8
%     subplot(2,4,i);
%     imshow(eightFaces(:,:,i), []);
% end
% saveas(gcf, './output/ps5-1-e.png');

%% Part 2

%% Part 2.a

W_training = U'*A;

%% Part 2.b

testFolder = dir("input/test");
testFile = {testFolder([testFolder.isdir]).name};
testFile = testFile(3:length(testFile));
T_test = zeros(10304,length(testFile)*2);

count = 1;

for i = 2:length(testFile)
    
    pgm = dir ("input/test/" +testFile(i));
    pgm = {pgm.name};
    pgm = pgm(3:length(pgm));
    
    for j = 1:2
        
        destination = "input/test/" + testFile(i) + "/" +pgm(j);
        tempIMG = imread(destination);
        tempIMG = reshape(tempIMG,[10304,1]);
        T_test(:,count) = tempIMG;
        
        count = count +1;
        
    end
end
T_test(:, 81:end) = [];

m_test = mean(T_test,2);

A_test = T_test - m_test;
C_test = A_test*A_test';

eigenValues_test = eig(A_test'*A_test);

eigenvaluesDescending_test = sort(eigenValues_test, 'descend');

i_test = 0;
v_test = 0;

denominator_test = 0;
for n_test = 1:80
    denominator_test = denominator_test + eigenvaluesDescending_test(n_test);
end
  
while v_test < 0.95
    
    i_test = i_test + 1;
    
    numerator_test = 0;

    for k_test = 1:i_test
        numerator_test = numerator_test + eigenvaluesDescending_test(k_test);
    end
    
    if i_test == 1
        v_test(1) = numerator_test / denominator_test;
    else
        v_test(end+1) = numerator_test / denominator_test;
    end

end
        
[U_test,D_test] = eigs(C_test,k_test);

W_testing = U_test'*A_test;

