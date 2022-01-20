
%% 
tic;
close all;
clear all;
clc;
format compact;
%%

data = imread('add_c.png');
figure;
imshow(data);
title('待处理图片')
%% 确定训练集
TrainData_bg = zeros(20,3,'double');
TrainData_fg = zeros(20,3,'double');
% 背景采样
msgbox('Please get 20 background samples','Background Samples','help');
pause;
for run = 1:20
    [x,y] = ginput(1);
    hold on;
    plot(x,y,'r*');
    x = uint8(x);
    y = uint8(y);
    TrainData_bg(run,1) = data(x,y,1);
    TrainData_bg(run,2) = data(x,y,2);
    TrainData_bg(run,3) = data(x,y,3);
end 
% 待分割出来的前景采样
msgbox('Please get 20 foreground samples which is the part to be segmented','Foreground Samples','help');
pause;
for run = 1:20
    [x,y] = ginput(1);
    hold on;
    plot(x,y,'ro');
    x = uint8(x);
    y = uint8(y);
    TrainData_fg(run,1) = data(x,y,1);
    TrainData_fg(run,2) = data(x,y,2);
    TrainData_fg(run,3) = data(x,y,3);
end 

TrainLabel = [zeros(length(TrainData_bg),1);ones(length(TrainData_fg),1)];
%% 建立支持向量机 基于libsvm
TrainData = [TrainData_bg;TrainData_fg];
model = svmtrain(TrainLabel, TrainData, '-t 1 -d 1');
%% 进行预测 i.e.进行图像分割 基于libsvm
preTrainLabel = svmpredict(TrainLabel, TrainData, model);
[m,n,k] = size(data);
TestData = double(reshape(data,m*n,k));
TestLabal = svmpredict(zeros(length(TestData),1), TestData, model);
%% 
ind = reshape([TestLabal,TestLabal,TestLabal],m,n,k);
ind = logical(ind);
data_seg = data;
data_seg(~ind) = 0;
figure;
imshow(data_seg);
figure;
subplot(1,2,1);
imshow(data);
title('原始图像');
subplot(1,2,2);
imshow(data_seg);
title('分割出的目标');
%%
toc;


