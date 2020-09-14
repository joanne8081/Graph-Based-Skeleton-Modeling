function plotConfMat(ConfMat, className, option)
%%% Input:
%%% ConfMat: NxN matrix , N=# classes
%%% className: 1xN cell array with the N class names
%%% option: 1:ratio; 2:#instances

figure;
classNum = size(ConfMat,1);

% mat = rand(5);           %# A 5-by-5 matrix of random values from 0 to 1
% mat(3,3) = 0;            %# To illustrate
% mat(5,2) = 0;            %# To illustrate
imagesc(ConfMat);            %# Create a colored plot of the matrix values
colormap(flipud(gray));  %# Change the colormap to gray (so higher values are
                         %#   black and lower values are white)
if option==1
    textStrings = num2str(ConfMat(:),'%0.2f');  %# Create strings from the matrix values
elseif option==2
    textStrings = num2str(ConfMat(:),'%d');  %# Create strings from the matrix values
end
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding

%% ## New code: ###
if option==1
    idx = find(strcmp(textStrings(:), '0.00'));
elseif option==2
    idx = find(strcmp(textStrings(:), '0'));
end
textStrings(idx) = {'   '};
%% ################

[x,y] = meshgrid(1:classNum);   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
                'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(ConfMat(:) > midValue,1,3);  %# Choose white or black for the
                                             %#   text color of the strings so
                                             %#   they can be easily seen over
                                             %#   the background color
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors

set(gca,'XTick',1:classNum,...                         %# Change the axes tick marks
        'XTickLabel',className,...  %#   and tick labels
        'YTick',1:classNum,...
        'YTickLabel',className,...
        'TickLength',[0 0]);
axis equal tight
xticklabel_rotate([],45)
    
end