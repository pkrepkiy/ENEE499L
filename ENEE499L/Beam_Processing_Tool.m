
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
    if isnan(thresholdMultiplier) || (thresholdMultiplier >= 1) || (thresholdMultiplier <= 0)

        error('Invalid input. Please enter a valid number between 0 and 1.');
    end
end

% fprintf('\n
% 
% 
% edgeMarkerNum = input('Enter number of edge markers or enter to skip (default is 360): ');
% 
% if isempty(edgeMarkerNum)
%     disp('Using default value: 360');
%    edgeMarkerNum = 360;
% else
% 
%     % Check if the input is a valid integer
%     if isnan(edgeMarkerNum) || (mod(edgeMarkerNum,2) ~= 0) || (edgeMarkerNum < 2)
% 
%         error('Invalid input. Please enter a valid, even integer.');
%     end
% end

edgeMarkerNum = 360;


fprintf('\nChange the depth of search for object (search sensitivity).\nThis value searches N pixels past\nthe first pixel with a value lower than the threshold. Increase for beam\nimages that are less dense or have more holes.\n\n')

searchDepth = input('Enter search depth or enter to skip (default is 20): ');


if isempty(searchDepth)
    fprintf('\nUsing default value: 20\n');
   searchDepth = 20;
else

    % Check if the input is a valid integer
    if isnan(searchDepth) || (searchDepth <= 0)

        error('Invalid input. Please enter a valid number greater than 0');
    end
end


% TODO:


fprintf('\nDefine the threshold to reject objects that have less length\nalong X and Y than the reject value. Any object that has X or Y distance\nless than N will be rejected. Increase this value if there is a lot of\nbright FOD or artifacts. Decrease for slits or smaller beams.\n\n')

rejectThresholdX = input('Enter reject size threshold in X or enter to skip (default is 15): ');

if isempty(rejectThresholdX)
    fprintf('\nUsing default X reject value: 15\n\n');
    rejectThresholdX = 15;
else

    % Check if the input is a valid integer
    if isnan(searchDepth) || (searchDepth <= 0)

        error('Invalid input. Please enter a valid number greater than 0');
    end
end

rejectThresholdY = input('Enter reject size threshold in Y or enter to skip (default is 15): ');

if isempty(rejectThresholdY)
    fprintf('\nUsing default Y reject value: 15\n');
    rejectThresholdY = 15;
else

    % Check if the input is a valid integer
    if isnan(searchDepth) || (searchDepth <= 0)

        error('Invalid input. Please enter a valid number greater than 0');
    end
end

calY = 0.0566; % VERTICAL CALIBRATION in mm/pixel

calX = 0.0566; % HORIZONTAL CALIBRATION in mm/pixel







fprintf('\n')
disp(['Searching for files in working directory ' pwd ' ...'])


%
% Search all files in the working directory
%

count = 0;

for i = 1:length(files)

    currFileName = files(i).name;
    [~, ~, fileExtension] = fileparts(currFileName);
    
    if strcmpi(fileExtension, '.tif')

        count = count +1;
        
        image = imread(currFileName);
        grayImage = im2gray(image);
        
        fprintf('\n')
        disp(['Processing image ' num2str(currFileName) ' ...'])

    
        %
        % Find max and set the initial cropped image
        %
        [maxIntensity, linearIndex] = max(grayImage(:));
        [yMaxCoord, xMaxCoord] = ind2sub(size(grayImage), linearIndex);



        %
        % Reject maximum intensities that are within 5 pixels of the border
        %
        if (yMaxCoord < 5 || yMaxCoord > size(gray,1) - 5) || (xMaxCoord < 5 || xMaxCoord > size(grayImage,2) - 5)

            
            %
            % Define a temporary image to find a lower maximum (bright FOD
            % rejection)
            %
        
            %
            % Create gaussian filter
            %
            sigma = 1;
            h = fspecial('gaussian', [5 5], sigma);
        
            %
            % Create a temporary smoothed image to reject bright FOD
            %
            smoothedImage = imfilter(grayImage, h, 'replicate');
        
            %
            % Keep decreasing the "maximum" value until larger objects are found 
            %
            while (yMaxCoord < 5 || yMaxCoord > size(grayImage,1) - 5) || (xMaxCoord < 5 || xMaxCoord > size(grayImage,2) - 5)
            
           
                smoothedImage(yMaxCoord,xMaxCoord) = NaN;
        
                [maxIntensity, linearIndex] = max(smoothedImage(:));
        
                [yMaxCoord, xMaxCoord] = ind2sub(size(smoothedImage), linearIndex);
        
            end
        end
        


        
        binaryMask = grayImage < maxIntensity- (maxIntensity*thresholdMultiplier);
    
        blackImage = grayImage;
        
        blackImage(binaryMask) = 0;  



        %
        % Set gain threshold for filter and number of edge markers
        %
        threshold = maxIntensity-(maxIntensity*thresholdMultiplier);


        edgeCoords = createEdgeMarkers(xMaxCoord, yMaxCoord, blackImage, threshold, edgeMarkerNum,searchDepth);



        %
        % Get min and max coordinates of the edge outline
        %
        minObjectX = min(edgeCoords(:,1));
        maxObjectX = max(edgeCoords(:,1));
        minObjectY = min(edgeCoords(:,2));
        maxObjectY = max(edgeCoords(:,2));
        
        %
        % Object / max Intensity rejection
        %
       


        %
        % Reject beam objects that are smaller than a particular length in
        % X and Y, 25 by default
        %
        if ((maxObjectX - minObjectX) < rejectThresholdX) || ((maxObjectY - minObjectY) < rejectThresholdY)
        
        
            %
            % Define a temporary image to find a lower maximum (bright FOD
            % rejection)
            %
        
            %
            % Create gaussian filter
            %
            sigma = 1;
            h = fspecial('gaussian', [5 5], sigma);
        
            %
            % Create a temporary smoothed image to reject bright FOD
            %
            smoothedImage = imfilter(grayImage, h, 'replicate');
        
            %
            % Keep decreasing the "maximum" value until larger objects are found 
            %
            while ((maxObjectX - minObjectX) < rejectThresholdX) || ((maxObjectY - minObjectY) < rejectThresholdY)
        
                %
                % Set the current maximum to NaN to find a new max.
                %
                smoothedImage(yMaxCoord,xMaxCoord) = NaN;
        
                [maxIntensity, linearIndex] = max(smoothedImage(:));
        
                threshold = maxIntensity-(maxIntensity*thresholdMultiplier);
        
        
                [yMaxCoord, xMaxCoord] = ind2sub(size(smoothedImage), linearIndex);
        
                edgeCoords = createEdgeMarkers(xMaxCoord, yMaxCoord, blackImage, threshold, edgeMarkerNum,searchDepth);
        
        
                %
                % Get new min and max coordinates of the edge outline
                %
                minObjectX = min(edgeCoords(:,1));
                maxObjectX = max(edgeCoords(:,1));
                minObjectY = min(edgeCoords(:,2));
                maxObjectY = max(edgeCoords(:,2));
        
            end
        
        end

        [X,Y]=meshgrid(1:size(blackImage,2), 1:size(blackImage,1));


        inside = inpolygon(X(:), Y(:), edgeCoords(:,1), edgeCoords(:,2));        


        % Calculate the Geometric centroid in X and Y
        GeometricCentroidX = mean(X(inside));
        GeometricCentroidY = mean(Y(inside));


        %
        % Create new edge coordinates based on the geometric centroid (to
        % prevent shadows)
        %

        edgeCoords = createEdgeMarkers(GeometricCentroidX, GeometricCentroidY, blackImage, threshold, edgeMarkerNum,searchDepth);
        

        [X,Y]=meshgrid(1:size(blackImage,2), 1:size(blackImage,1));


        % Find which points are inside the object's outline
        inside = inpolygon(X(:), Y(:), edgeCoords(:,1), edgeCoords(:,2));
        

        %
        % Calculate NEW geometric centroids
        %

        GeometricCentroidX = mean(X(inside));
        GeometricCentroidY = mean(Y(inside));

        % This creates a logical array- points inside the object are a "1"
        % and points outside the object are a "0"
        propInside = reshape(inside,size(blackImage,1),size(blackImage,2));

        % Mask the original image with the logical array
        cropImage = uint32(grayImage) .* uint32(propInside);


        % Set zero values to NaN to be omitted from intensity centroid
        % calculation
        cropImage(cropImage == 0) = NaN;

        minObjectX = min(edgeCoords(:,1));
        maxObjectX = max(edgeCoords(:,1));
        minObjectY = min(edgeCoords(:,2));
        maxObjectY = max(edgeCoords(:,2));



        % Total pixel intensity sums
        I_0 = sum(sum(cropImage(:,:)));


        %% Calculate first moment in X and Y

        % Create vectors
        xList = uint32(1:size(cropImage,2));

        yList = (uint32(1:size(cropImage,1)))';

        yVec = round(minObjectY):round(maxObjectY);

        xVec = round(minObjectX):round(maxObjectX);

        XMoment = sum(sum(cropImage(yVec,:) .* xList,2))/I_0;

        YMoment = sum(sum(cropImage(:,xVec) .* yList,1))/I_0;


        %% Calculate first moment in Y

        %
        % Calculate Moment in Y
        %

        %
        % Initialize moment in Y and iteration variable
        %
        YMoment = 0;
        iter = 0;


        %
        % Move/scan down the image vertically
        %
        for j = round(minObjectX):round(maxObjectX)

            %
            % Multiply each Y index with the binary mask at that particular column,
            % Sum all Y.
            % (Here we are adding up all of the Y coordinates that are
            % included in the cropped beam image to calculate the first
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
                iter = iter + 1;

                
            end

        end

        %
        % Divide by number of non-zero iterations 
        %
        YMoment = YMoment / iter;



        %% Calculate second moment in X

        %
        % Calculate second moment in X
        %

        %
        % Initialize moment in X and iteration variable
        %
        X2Moment = 0;
        iter = 0;

        %
        % Move/scan down the image vertically
        %
        for j = round(minObjectY):round(maxObjectY)

            % Get coordinates of all X value inside the given row
            X2Coord = uint16((1:size(cropImage,2))) .* uint16(propInside(j,:));

            % Sum X coordinates squared with respective values at those
            % coordinates in the row, divide by the sum of values in that
            % row

            Row2MomentX = sum(uint32(X2Coord.^2).* uint32(cropImage(j,:)))/sum(cropImage(j,:).* uint16(propInside(j,:)));


            if ~isnan(Row2MomentX)

                X2Moment = X2Moment + Row2MomentX;
                iter = iter + 1;

            end

        end

        %
        % Divide by number of non-zero iterations and take square root
        % (for second moment)
        %
        X2Moment = sqrt(X2Moment) / iter;

        %% Calculate second moment in Y

        %
        % Calculate second moment in Y
        %

        %
        % Initialize moment in Y and iteration variable
        %
        Y2Moment = 0;
        iter = 0;


        %
        % Move/scan down the image vertically
        %
        for j = round(minObjectX):round(maxObjectX)

            %
            % Multiply each Y index with the binary mask at that particular column,
            % Sum all Y.
            % (Here we are adding up all of the Y coordinates that are
            % included in the cropped beam image to calculate the first
            % moment)
            % Vertical arrays
            %
            Y2Coord = uint16(((1:size(cropImage,1))')) .* uint16(propInside(:,j));


            %
            % This gives the yCoordinate for the moment of that row
            %
            Col2MomentY = sum(uint32(Y2Coord.^2).* uint32(cropImage(:,j)))/sum(cropImage(:,j).* uint16(propInside(:,j)));


            %
            % Check if NOT DIV0
            %
            if ~isnan(Col2MomentY)


                %
                % Add up all X Moment coordinates for averaging later
                %
                Y2Moment = Y2Moment + Col2MomentY;
                iter = iter + 1;

                
            end

        end

        %
        % Divide by number of non-zero iterations and take square root
        % (for second moment)
        %
        Y2Moment = sqrt(Y2Moment) / iter;





        %% Plot results

        %
        % Plotting
        %

        figure(i)
        % subplot(1,2,1), 
        
        imshow(grayImage), title(num2str(currFileName))
    
        hold on

        plot(xMaxCoord,yMaxCoord, 'k*', 'MarkerSize', 25);

        plot(GeometricCentroidX,GeometricCentroidY,'b+','MarkerSize',25);

        plot(XMoment, YMoment,'gx','MarkerSize',25)


        plot(X2Moment, Y2Moment,'mv','MarkerSize',25)

        % text(50, 50, ['First Moment in X: ' num2str(XMoment)], 'Color', 'red', 'FontSize', 10);
        % text(50, 30, ['First Moment in Y: ' num2str(YMoment)], 'Color', 'red', 'FontSize', 10);
        % 
        % text(50, 90, ['Second Moment in X: ' num2str(X2Moment)], 'Color', 'red', 'FontSize', 10);
        % text(50, 70, ['Second Moment in Y: ' num2str(Y2Moment)], 'Color', 'red', 'FontSize', 10);
        
        fprintf('\n')
        disp(['First Moment in X: ' num2str(XMoment)])
        disp(['First Moment in Y: ' num2str(YMoment)])
        disp(['Second Moment in Y: ' num2str(Y2Moment)])
        disp(['Second Moment in Y: ' num2str(Y2Moment)])

        plot(edgeCoords(:, 1), edgeCoords(:, 2), 'ro-','MarkerSize',3);

%[num2str(edgeMarkerNum) ,' Edge Markers']

        % hold off;

        % subplot(1,2,2), imshow(blackImage), title('Object Processing')

        % hold on

        plot(minObjectX,minObjectY,'cd')
        plot(maxObjectX,maxObjectY,'cd')
        plot(minObjectX,maxObjectY,'cd')
        plot(maxObjectX,minObjectY,'cd')


        legend('Peak intensity','Geometric Centroid','First Moment','Second Moment')


        % plot(X(inside),Y(inside),'rd','MarkerSize',1)

        % legend('Inner object grid')
        hold off
    
    end

end



function outCoords = createEdgeMarkers(xCoord, yCoord, blackImage, threshold, numMarkers,searchDepth)


        % Iterate over angles to evenly divide the circle
        
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
            while ((newCoord(2) + searchDepth*dy <= size(blackImage,1) && newCoord(2) + searchDepth*dy > 1)) && ((newCoord(1) + searchDepth*dx <= size(blackImage,2)) && (newCoord(1) + searchDepth*dx > 1))

                % Check current position
                if blackImage(round(newCoord(2)), round(newCoord(1))) < threshold

                    % Check up to 20 indices in front
                    checkVar = newCoord + [dx, dy];
                    count = 1;

                    while count < searchDepth && blackImage(round(checkVar(2)), round(checkVar(1))) < threshold
                        checkVar = checkVar + [dx, dy];
                        count = count + 1;
                    end

                    if count == searchDepth
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


