
% Author: Peter Krepkiy (pkrepkiy@umd.edu, krepkiyp@yahoo.com), 

% Last edit: February 12, 2024

% Revision: 0

% Description:

% This program reads .tif images for the purposes of extracting information
% that is useful for research efforts with regard to the Long Solenoid
% Experiment (LSE) at the University of Maryland Institute for Research in
% Electronics and Applied Physics


%
% Establish directories and control variables
%

close all; clear; clc

warning('off','all')

files = dir(fullfile(pwd, '*.*'));



fprintf('Set the threshold multiplier. The algorithm searches for pixels that \nare lower than this threshold to form the beam object. The formula is like so: \nthreshold = maxIntensity - maxIntensity*thresholdMultiplier. \nThis value must be between 0 and 1.\n\n');

thresholdMultiplier = input('Enter intensity cutoff threshold or enter to skip (default is 1/sqrt(2)): ');

if isempty(thresholdMultiplier)
    fprintf('\nUsing default value: 1/sqrt(2)\n');
   thresholdMultiplier = 1/sqrt(2);
else

    % Check if the input is a valid integer
    if isnan(thresholdMultiplier) || (thresholdMultiplier > 1) || (thresholdMultiplier < 0)

        error('Invalid input. Please enter a valid number between 0 and 1.');
    end
end

fprintf('\nChange number of edge markers as needed, must be an even number (recommended 360)\nThe object outline is formed out of a border of N radially distributed edge markers.\n\n')


edgeMarkerNum = input('Enter number of edge markers or enter to skip (default is 360): ');

if isempty(edgeMarkerNum)
    disp('Using default value: 360');
   edgeMarkerNum = 360;
else

    % Check if the input is a valid integer
    if isnan(edgeMarkerNum) || (mod(edgeMarkerNum,2) ~= 0) || (edgeMarkerNum < 2)
       
        error('Invalid input. Please enter a valid, even integer.');
    end
end


% TODO:


%
% Change the depth of search for object. This value searches N pixels past
% the first pixel with a value lower than the threshold. Increase for beam
% images that are less dense or have more holes.
%
% searchDepth = searchDepth.Value ;  %20;


%
% Define the threshold to reject objects that have less length
% along X and Y than the reject value. Any object that has X or Y distance
% less than N will be rejected. Increase this value if there is a lot of
% bright FOD or artifacts.
% %
% rejectThresholdX = rejectThresholdX.Value ; %25;
% rejectThresholdY = rejectThresholdY.Value ; %25;





fprintf('\n')
disp(['Searching for files in working directory ' pwd ' ...'])


%
% Search all files in the working directory
%
for i = 1:length(files)

    currFileName = files(i).name;
    [~, ~, fileExtension] = fileparts(currFileName);
    
    if strcmpi(fileExtension, '.tif')
        
        image = imread(currFileName);
        grayImage = im2gray(image);
        filteredImage = medfilt2(grayImage, [3, 3]);
        edgeImage = edge(filteredImage, 'Canny');
        
        
        %
        % Find coordinates and value of the maximum in the image
        % TODO: Expand this to search for largest contiguous object
        % to filter out FOD and random noise
        %

        fprintf('\n')
        disp(['Processing image' num2str(currFileName) ' ...'])

        [blackImage, xMaxCoord, yMaxCoord, maxIntensity] = fiducials(grayImage);


        %
        % Set gain threshold for filter and number of edge markers
        %
        % threshold = maxIntensity-(maxIntensity/sqrt(2)); % RMS

        threshold = maxIntensity-(maxIntensity*thresholdMultiplier);

        %
        % Change number of edge markers as needed, up to 360
        %
        % edgeMarkerNum = 360;



        edgeCoords = createEdgeMarkers(xMaxCoord, yMaxCoord, blackImage, threshold, edgeMarkerNum);

        [X,Y]=meshgrid(1:size(blackImage,2), 1:size(blackImage,1));


        inside = inpolygon(X(:), Y(:), edgeCoords(:,1), edgeCoords(:,2));        


        % Calculate the Geometric centroid in X and Y
        GeometricCentroidX = mean(X(inside));
        GeometricCentroidY = mean(Y(inside));

        propInside = reshape(inside,size(blackImage,1),size(blackImage,2));

        cropImage = grayImage .* uint16(propInside);


        % Set zero values to NaN to be omitted from intensity centroid
        % calculation
        cropImage(cropImage == 0) = NaN;

        minObjectX = min(edgeCoords(:,1));
        maxObjectX = max(edgeCoords(:,1));
        minObjectY = min(edgeCoords(:,2));
        maxObjectY = max(edgeCoords(:,2));



        %
        % Calculate moment in X
        %


        %
        % Initialize moment in X
        %
        XMoment = 0;


        %
        % Move/scan down the image vertically
        %
        for j = round(minObjectY):round(maxObjectY)

            %
            % Multiply each X index with the binary mask at that particular row,
            % Sum all X.
            % (Here we are adding up all of the X coordinates that are
            % included in the cropped beam image to calculate the second
            % moment)
            %
            XCoord = uint16((1:size(cropImage,2))) .* uint16(propInside(j,:));


            %
            % This gives the xCoordinate for the moment of that row
            %
            RowMomentX = sum(uint32(XCoord).* uint32(cropImage(j,:)))/sum(cropImage(j,:).* uint16(propInside(j,:)));


            %
            % Check if NOT DIV0
            %
            if ~isnan(RowMomentX)


                %
                % Add up all X Moment coordinates for averaging later
                %
                XMoment = XMoment + RowMomentX;

                
            end

        end

        %
        % Divide by number of iterations 
        %
        XMoment = XMoment / size(round(minObjectY):round(maxObjectY),2);


        %
        % Calculate Moment in Y
        %

        %
        % Initialize moment in X
        %
        YMoment = 0;


        %
        % Move/scan down the image vertically
        %
        for j = round(minObjectX):round(maxObjectX)

            %
            % Multiply each Y index with the binary mask at that particular column,
            % Sum all Y.
            % (Here we are adding up all of the Y coordinates that are
            % included in the cropped beam image to calculate the second
            % moment)
            % Vertical arrays
            %
            YCoord = uint16(((1:size(cropImage,1))')) .* uint16(propInside(:,j));


            %
            % This gives the xCoordinate for the moment of that row
            %
            ColMomentY = sum(uint32(YCoord).* uint32(cropImage(:,j)))/sum(cropImage(:,j).* uint16(propInside(:,j)));


            %
            % Check if NOT DIV0
            %
            if ~isnan(ColMomentY)


                %
                % Add up all X Moment coordinates for averaging later
                %
                YMoment = YMoment + ColMomentY;

                
            end

        end

        %
        % Divide by number of iterations 
        %
        YMoment = YMoment / size(round(minObjectX):round(maxObjectX),2);


        %
        % Plotting
        %

        figure(i)
        subplot(1,2,1), imshow(grayImage), title(num2str(currFileName))
    
        hold on

        plot(edgeCoords(:, 1), edgeCoords(:, 2), 'ro-','MarkerSize',3);

        plot(xMaxCoord,yMaxCoord, 'k*', 'MarkerSize', 25);

        plot(GeometricCentroidX,GeometricCentroidY,'b+','MarkerSize',25);

        plot(XMoment, YMoment,'gx','MarkerSize',25)

        text(50, 50, ['Moment in X: ' num2str(XMoment)], 'Color', 'red', 'FontSize', 10);
        text(50, 30, ['Moment in Y: ' num2str(YMoment)], 'Color', 'red', 'FontSize', 10);


        legend([num2str(edgeMarkerNum) ,' Edge Markers'],'Peak intensity','Geometric Centroid','Moment Centroid')

        hold off;

        subplot(1,2,2), imshow(cropImage), title('Object Processing')

        hold on
        plot(X(inside),Y(inside),'rd','MarkerSize',1)

        legend('Inner object grid')
        hold off


        end

    end

%
% Uncomment for video animation:
%
% end



function [blackImage, xCoord, yCoord, maxIntensity] = fiducials(grayImage)


    [maxIntensity, linearIndex] = max(grayImage(:));
    [yCoord, xCoord] = ind2sub(size(grayImage), linearIndex);
    
    binaryMask = grayImage < maxIntensity- (maxIntensity/sqrt(2));

    blackImage = grayImage;
    
    blackImage(binaryMask) = 0;  

end


function outCoords = createEdgeMarkers(xCoord, yCoord, blackImage, threshold, numMarkers)

        %
        % Iterate over angles to evenly divide the circle
        %
        angle = linspace(0, 360, numMarkers + 1);
        angle = angle(1:end-1);  % Exclude the last element to avoid duplicates

        outCoords = zeros(numMarkers,2);

        for i= 1:numMarkers

            % Compute x and y components based on the angle
            dx = cosd(angle(i));
            dy = sind(angle(i));

            % Initialize variables
           newCoord = [xCoord, yCoord] +  [dx, dy];
    

            % Loop to move along the direction until the threshold condition is met
            while (newCoord(2) + 10*dy <= size(blackImage,1)) && (newCoord(1) + 10*dx <= size(blackImage,2))

                % Check current position
                if blackImage(round(newCoord(2)), round(newCoord(1))) < threshold

                    % Check up to 20 indices in front
                    checkVar = newCoord + [dx, dy];
                    count = 1;

                    while count < 20 && blackImage(round(checkVar(2)), round(checkVar(1))) < threshold
                        checkVar = checkVar + [dx, dy];
                        count = count + 1;
                    end

                    if count == 20
                        break;
                    end
                end

              
                count = 0;
                newCoord = newCoord +  [dx, dy];

                
            end

            % Reset coordinates for the next iteration
            outCoords(i,:) = newCoord;
        end
end


