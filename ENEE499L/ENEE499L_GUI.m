%ENEE499L

%Peter Krepkiy

% February 4, 2024

% Revision 0

% Description:

% This program reads .tif images for the purposes of extracting information
% that is useful for research efforts with regard to the Long Solenoid
% Experiment (LSE) at the University of Maryland Institute for Research in
% Electronics and Applied Physics

% The program will generate a GUI to adjust parameters automatically.

all_fig = findall(0, 'type', 'figure');
close(all_fig)

close all; clear; clc

warning('off','all')


%
% Create UI
%

% Create the main UI figure
mainFig = uifigure('Name', 'Beam Image Processor', 'Position', [200, 100, 1000, 600]);



% Create sliders
thresholdMultiplier = uislider(mainFig,'Position',[50, 450, 200, 30], 'Limits', [0, 1],'Value',0.5);
searchDepth = uislider(mainFig,'Position', [50, 350, 200, 30], 'Limits', [0, 50],'Value',20);
rejectThresholdX = uislider(mainFig,'Position', [50, 250, 200, 30], 'Limits', [0, 100],'Value',25);
rejectThresholdY = uislider(mainFig,'Position', [50, 150, 200, 30], 'Limits', [0, 100],'Value',25);


% Add a listener to sliders for real-time updates
SL1 = addlistener(thresholdMultiplier, 'ValueChanged', @(src, event) updateOutput());
SL2 = addlistener(searchDepth, 'ValueChanged', @(src, event) updateOutput());
SL3 = addlistener(rejectThresholdX, 'ValueChanged', @(src, event) updateOutput());
SL4 = addlistener(rejectThresholdY, 'ValueChanged', @(src, event) updateOutput());



% Make UI figure
ax=uiaxes(mainFig,'Position', [350, 100, 640, 480]);


uilabel(mainFig,'Text','Threshold Multiplier','Position',[50 420 200 100]);
uilabel(mainFig,'Text','Search Depth','Position',[50 320 200 100]);
uilabel(mainFig,'Text','X Reject Threshold','Position',[50 220 200 100]);
uilabel(mainFig,'Text','Y Reject Threshold','Position',[50 120 200 100]);


refreshButton = uibutton(mainFig, 'push', 'Position', [50, 50, 100, 30], 'Text', 'Refresh', 'ButtonPushedFcn', @(src, event) refreshButtonCallback());



doBeamAlgorithm(ax,thresholdMultiplier.Value,searchDepth.Value,rejectThresholdX.Value,rejectThresholdY.Value)


pause(5)

while ishandle(mainFig)

    updateOutput
    
    if ~ishandle(mainFig)

        closereq
        return;
    end

end


%
% Set parameters
%


%
% Change the depth of search for object. This value searches N pixels past
% the first pixel with a value lower than the threshold. Increase for beam
% images that are less dense or have more holes.
%
% searchDepth = searchDepth.Value ;  %20;


%
% Change number of edge markers as needed, must be an even number
% (recommened 360). The object outline is formed out of a border of N
% radially distributed edge markers.
%
% edgeMarkerNum = 360;


%
% Define the threshold to reject objects that have less length
% along X and Y than the reject value. Any object that has X or Y distance
% less than N will be rejected. Increase this value if there is a lot of
% bright FOD or artifacts.
% %
% rejectThresholdX = rejectThresholdX.Value ; %25;
% rejectThresholdY = rejectThresholdY.Value ; %25;



%
% Set the threshold multiplier. The algorithm searches for pixels that are
% lower than this threshold to form the beam object. The formula is like so:
% threshold = maxIntensity - maxIntensity*thresholdMultiplier
% This value must be between 0 and 1.
%
% thresholdMultiplier = 0.4;




function doBeamAlgorithm(ax,thresholdMultiplier,searchDepth,rejectThresholdX,rejectThresholdY)
%
% Set filepath to current working directory.
%

files = dir(fullfile(pwd, '*.*'));

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
        [blackImage, xMaxCoord, yMaxCoord, maxIntensity] = fiducials(grayImage,thresholdMultiplier);


        %
        % Set gain threshold for filter and number of edge markers
        %

        % if thresholdMultiplier > 1 || thresholdMultiplier < 0
        % 
        %     disp('Error: Threshold multiplier value must be between 0 and 1')
        %     return;
        % 
        % end

        threshold = maxIntensity-(maxIntensity*thresholdMultiplier); % RMS

        edgeMarkerNum = 360;


        %
        % Find outer edge markers given a maximum point, number of edge
        % Markers, and gain threshold
        %

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
        % X and Y, OR if the max Intensity is within 5 pixels of the border
        %
        if (yMaxCoord < 5 || yMaxCoord > size(blackImage,1) - 5) || (xMaxCoord < 5 || xMaxCoord > size(blackImage,2) - 5) || (((maxObjectX - minObjectX) < rejectThresholdX) || ((maxObjectY - minObjectY) < rejectThresholdY))

           
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
            while (yMaxCoord < 5 || yMaxCoord > size(blackImage,1) - 5) || (xMaxCoord < 5 || xMaxCoord > size(blackImage,2) - 5) || ((maxObjectX - minObjectX) < rejectThresholdX) || ((maxObjectY - minObjectY) < rejectThresholdY)

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


        % Check which points are inside the object's outline
        inside = inpolygon(X(:), Y(:), edgeCoords(:,1), edgeCoords(:,2));
        

        % Calculate the Geometric centroid in X and Y
        GeometricCentroidX = mean(X(inside));
        GeometricCentroidY = mean(Y(inside));

        %
        % Create new edge coordinates based on the geometric centroid (to
        % prevent shadows)
        %

        edgeCoords = createEdgeMarkers(GeometricCentroidX, GeometricCentroidY, blackImage, threshold, edgeMarkerNum,searchDepth);
        

        %
        % Get min and max coordinates of the edge outline
        %
        minObjectX = min(edgeCoords(:,1));
        maxObjectX = max(edgeCoords(:,1));
        minObjectY = min(edgeCoords(:,2));
        maxObjectY = max(edgeCoords(:,2));

        [X,Y]=meshgrid(1:size(blackImage,2), 1:size(blackImage,1));


        % Check which points are inside the object's outline
        inside = inpolygon(X(:), Y(:), edgeCoords(:,1), edgeCoords(:,2));
        

        %
        % Calculate NEW geometric centroids
        %

        GeometricCentroidX = mean(X(inside));
        GeometricCentroidY = mean(Y(inside));





        propInside = reshape(inside,size(blackImage,1),size(blackImage,2));

        cropImage = grayImage .* uint16(propInside);


        % Set zero values to NaN to be omitted from intensity centroid
        % calculation
        cropImage(cropImage == 0) = NaN;


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

        
        imshow(grayImage,'Parent',ax), title(num2str(currFileName),'Parent',ax)
    
        hold(ax);

        plot(edgeCoords(:, 1), edgeCoords(:, 2), 'ro-','MarkerSize',3,'Parent',ax);

        plot(xMaxCoord,yMaxCoord, 'k*', 'MarkerSize', 25,'Parent',ax);

        plot(GeometricCentroidX,GeometricCentroidY,'b+','MarkerSize',25,'Parent',ax);

        plot(XMoment, YMoment,'gx','MarkerSize',25,'Parent',ax)

        text(40, 50, ['Moment in X: ' num2str(XMoment)], 'Color', 'red', 'FontSize', 20,'Parent',ax);
        text(40, 20, ['Moment in Y: ' num2str(YMoment)], 'Color', 'red', 'FontSize', 20,'Parent',ax);

        legend(ax,[num2str(edgeMarkerNum) ,' Edge Markers'],'Peak intensity','Geometric Centroid','Moment Centroid')

        hold(ax)

%         subplot(1,2,2), imshow(ax,cropImage), title('Object Processing')

%         hold on
%         plot(ax,X(inside),Y(inside),'rd','MarkerSize',1)
% 
%         legend('Inner object grid')
%         hold off


        end

    end

end

function [blackImage, xCoord, yCoord, maxIntensity] = fiducials(grayImage,thresholdMultiplier)


    [maxIntensity, linearIndex] = max(grayImage(:));
    [yCoord, xCoord] = ind2sub(size(grayImage), linearIndex);
    
    binaryMask = grayImage < maxIntensity- (maxIntensity*thresholdMultiplier);

    blackImage = grayImage;
    
    blackImage(binaryMask) = 0;  

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

function updateOutput(~, ~)
    
    drawnow;

end

% Callback function for refresh button click
function refreshButtonCallback(~, ~)
    % Re-run doBeamAlgorithm with updated slider values
    doBeamAlgorithm(ax, thresholdMultiplier.Value, searchDepth.Value, rejectThresholdX.Value, rejectThresholdY.Value);
end
