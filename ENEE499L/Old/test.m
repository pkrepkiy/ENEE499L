  



% close all; clear; clc


    image = imread('picture1.tif');
    

    grayImage = im2gray(image);

    figure(1)

    [maxIntensity, linearIndex] = max(grayImage(:));
    [yCoord, xCoord] = ind2sub(size(grayImage), linearIndex);
    
    binaryMask = grayImage < maxIntensity/sqrt(2);
    
    blackImage = grayImage;
    
    blackImage(binaryMask) = 0;  

    figure(1)
    plot3DArray(grayImage)

function plot3DArray(array)
    % Check if the input is a 3D array
    if ndims(array) ~= 3
        error('Input must be a 3D array.');
    end

    % Get the size of the array
    [rows, cols, depth] = size(array);

    % Create a 3D plot
    figure;
    hold on;
    
    % Plot each slice in the Z direction
    for z = 1:depth
        slice = array(:, :, z);
        [x, y] = meshgrid(1:cols, 1:rows);
        plot3(x(:), y(:), z * ones(size(x(:))), 'b.'); % Change 'b.' to customize the plot style
    end

    % Customize the plot
    title('Intensity plot');
    xlabel('X-axis');
    ylabel('Y-axis');
    zlabel('Z-axis');
    grid on;
    hold off;
end


