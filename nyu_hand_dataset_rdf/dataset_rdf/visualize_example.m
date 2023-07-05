% Jonathan Tompson, New York University - 1/15/2015
% This is a simple script to visualize a training example
clearvars; close all; clc; rng(0);

dataset_dir = '.\dataset\';
image_index = 1;
filename_prefix = sprintf('%07d', image_index);

%% Load and display a depth image example
data = imread([dataset_dir, 'depth_', filename_prefix, '.png']);
depth = uint16(data(:,:,3)) + bitsll(uint16(data(:,:,2)), 8);
labels = data(:,:,1) > 0;

figure;
set(gcf, 'Position', [200 200 1200 400]);
subplot(1,2,1);
imshow(depth, [0, max(depth(:))]);
subplot(1,2,2);
imshow(labels);
