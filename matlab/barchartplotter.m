%% Bar Chart Plotter for CSV (InfluxDB-style timestamps)
% Author: Amarjith TK
% Purpose: Create bar charts for kWh and Power Factor from time-series data
clc; clear; close all;

%% ================== USER DEFINED INPUTS ==================
% File and data parameters
filename = 'resi.csv';  % CSV file name
parameters = {'kwh_15min', 'power_factor'};  % Columns to plot

% Date range (UTC timezone)
startDate = datetime('2025-01-01 06:00:00', 'TimeZone', 'UTC', 'InputFormat', 'yyyy-MM-dd HH:mm:ss');
endDate   = datetime('2025-01-02 06:00:00', 'TimeZone', 'UTC', 'InputFormat', 'yyyy-MM-dd HH:mm:ss');

intervalMinutes = [];  % Set to [] for no resampling (use raw data)

% Plot labels (USER CONTROLLED)
xAxisLabels = {'Time', 'Time'};
yAxisLabels = {'Energy Consumption (kWh)', 'Average Power Factor'};
plotTitles = {'Energy Consumption - 15 Min Intervals (Residential)', 'Average Power Factor Profile (Residential)'};

% Bar styling
barColors = {[0.8500 0.3250 0.0980], [0.4660 0.6740 0.1880]};  % Orange for kWh, Green for PF
edgeColor = 'none';       % Edge color ('none' for no edge, or [0 0 0] for black)
barWidth = 0.8;           % Bar width (0.8 = default, 1.0 = no gaps)
faceAlpha = 0.9;          % Transparency (0-1, 1=opaque)
gridOn = true;            % Grid on/off

% X-axis time format
xTickIntervalHours = 2;   % Interval for x-tick labels (in hours)
xTickFormat = 'HH:mm';    % Time format for x-axis
xTickRotation = 45;       % Rotation angle for x-axis labels

% Bar labels (values on top of bars)
showBarLabels = false;    % true to show values on bars
barLabelFormat = '%.2f';  % Format for bar labels (e.g., '%.2f' for 2 decimals)

% Font sizes
titleFontSize = 14;
axisFontSize = 12;
tickFontSize = 11;
barLabelFontSize = 9;

% Figure size
figureWidth = 800;   % Width in pixels
figureHeight = 500;  % Height in pixels

% Figure save options
saveFigures = true;  % Set to false to skip saving
savePrefix = 'bar_'; % Prefix for saved filenames
%% =========================================================

%% 1. Read dataset
data = readtable(filename);

% Detect timestamp column automatically
timeCol = find(strcmpi(data.Properties.VariableNames, 'timestamp') | ...
               strcmpi(data.Properties.VariableNames, 'time') | ...
               strcmpi(data.Properties.VariableNames, 'datetime'), 1);

if isempty(timeCol)
    error('Timestamp column not found. Ensure a column named "timestamp", "time", or "datetime" exists.');
end

% Convert InfluxDB-style timestamps → MATLAB datetime
data.Time = datetime(data{:, timeCol}, ...
    'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss''Z''', 'TimeZone', 'UTC');
data.Time.TimeZone = 'UTC';

%% 2. Filter by time range
startDate.TimeZone = 'UTC';
endDate.TimeZone = 'UTC';
mask = (data.Time >= startDate) & (data.Time <= endDate);
dataFiltered = data(mask, :);

if isempty(dataFiltered)
    error('No data found in the given date range.');
end

%% 3. Resample if interval specified
if ~isempty(intervalMinutes)
    numericVars = varfun(@isnumeric, dataFiltered, 'OutputFormat', 'uniform');
    dataFiltered = dataFiltered(:, numericVars | strcmp(dataFiltered.Properties.VariableNames, 'Time'));
    T = table2timetable(dataFiltered, 'RowTimes', 'Time');
    T_resampled = retime(T, 'regular', 'mean', 'TimeStep', minutes(intervalMinutes));
else
    T_resampled = table2timetable(dataFiltered, 'RowTimes', 'Time');
end

%% 4. Validate parameters exist
for i = 1:length(parameters)
    if ~ismember(parameters{i}, T_resampled.Properties.VariableNames)
        error('Parameter "%s" not found in dataset.', parameters{i});
    end
end

%% 5. Create separate bar charts
for i = 1:length(parameters)
    param = parameters{i};
    
    % Create new figure for each parameter
    fig = figure('Color', 'w', 'Position', [100, 100, figureWidth, figureHeight]);
    
    % Extract data for current parameter
    timeData = T_resampled.Time;
    paramData = T_resampled.(param);
    
    % Remove NaN values
    validIdx = ~isnan(paramData);
    timeData = timeData(validIdx);
    paramData = paramData(validIdx);
    
    % Create bar chart
    b = bar(timeData, paramData, barWidth);
    
    % Apply styling
    b.FaceColor = barColors{i};
    b.EdgeColor = edgeColor;
    b.FaceAlpha = faceAlpha;
    
    % Add bar labels if requested
    if showBarLabels
        xtips = b.XEndPoints;
        ytips = b.YEndPoints;
        labels = arrayfun(@(x) sprintf(barLabelFormat, x), paramData, 'UniformOutput', false);
        text(xtips, ytips, labels, 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', 'FontSize', barLabelFontSize);
    end
    
    % Apply grid
    if gridOn
        grid on;
    end
    
    % Set labels and title
    xlabel(xAxisLabels{i}, 'FontSize', axisFontSize, 'FontWeight', 'bold');
    ylabel(yAxisLabels{i}, 'FontSize', axisFontSize, 'FontWeight', 'bold');
    title(plotTitles{i}, 'FontSize', titleFontSize, 'FontWeight', 'bold');
    
    % Configure x-axis ticks
    if ~isempty(xTickIntervalHours)
        tickPositions = startDate:hours(xTickIntervalHours):endDate;
        xticks(tickPositions);
    end
    xtickformat(xTickFormat);
    
    % Remove the secondary date label (removes "Jan2025 - Jan 02, 2025")
    ax = gca;
    ax.XAxis.SecondaryLabel.String = '';
    
    % Rotate x-axis labels
    if xTickRotation ~= 0
        xtickangle(xTickRotation);
    end
    
    % Set tick font size
    set(gca, 'FontSize', tickFontSize);
    box on;
    
   
end

fprintf('\nAll figures saved successfully!\n');
