%% Time-Series Plotter for CSV (InfluxDB-style timestamps)
% Author: Amarjith TK
% Purpose: Filter and plot multiple time-series parameters with flexible date selection
clc; clear; close all;

%% ================== USER DEFINED INPUTS ==================
% File and data parameters
filename = 'resi.csv';  % CSV file name
parameters = {'active_power_kw', 'reactive_power_kvar', 'voltage', 'current'};  % Columns to plot

% Date range (UTC timezone)
startDate = datetime('2025-01-01 06:00:00', 'TimeZone', 'UTC', 'InputFormat', 'yyyy-MM-dd HH:mm:ss');
endDate   = datetime('2025-01-02 06:00:00', 'TimeZone', 'UTC', 'InputFormat', 'yyyy-MM-dd HH:mm:ss');
intervalMinutes = 60;  % Sampling interval (e.g., 15, 30, 60)

% Plot labels and titles (USER CONTROLLED) - one for each parameter
xAxisLabels = {'Time(24hrs)', 'Time', 'Time', 'Time'};
yAxisLabels = {'Active Power (kW)', 'Reactive Power (kVAR)', 'Voltage (V)', 'Current (A)'};
plotTitles = {'Active Power Profile - 24 Hour Cycle(Residential)', ...
              'Reactive Power Profile - 24 Hour Cycle(Residential)', ...
              'Voltage Profile - 24 Hour Cycle(Residential)', ...
              'Current Profile - 24 Hour Cycle(Residential)'};

% X-axis tick control
tickIntervalHours = 2;  % Tick interval in hours (2 hours spacing)
xTickFormat = 'HH:mm';  % Time format: 'HH:mm' shows 06:00, 08:00, etc.
xTickRotation = 0;      % Rotation angle for x-axis tick labels (0, 45, 90, etc.)

% Plot styling - Define colors for each parameter
lineColors = {
    [0 0.4470 0.7410],      % Blue for active power
    [0.8500 0.3250 0.0980], % Orange for reactive power
    [0.9290 0.6940 0.1250], % Yellow for voltage
    [0.4940 0.1840 0.5560]  % Purple for current
};
lineWidth = 1.5;
gridOn = true;  % true or false

% Font sizes
titleFontSize = 14;
axisFontSize = 12;
tickFontSize = 11;

% Figure size
figureWidth = 800;   % Width in pixels
figureHeight = 500;  % Height in pixels

% Figure save options
saveFigures = true;  % Set to false to skip saving
savePrefix = 'timeseries_';  % Prefix for saved filenames
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

% Force all timestamps to UTC to avoid timezone mismatch
data.Time.TimeZone = 'UTC';
startDate.TimeZone = 'UTC';
endDate.TimeZone   = 'UTC';

%% 2. Filter by time range
mask = (data.Time >= startDate) & (data.Time <= endDate);
dataFiltered = data(mask, :);

if isempty(dataFiltered)
    error('No data found in the given date range.');
end

%% 3. Keep only numeric columns for resampling
numericVars = varfun(@isnumeric, dataFiltered, 'OutputFormat', 'uniform');
dataFiltered = dataFiltered(:, numericVars | strcmp(dataFiltered.Properties.VariableNames, 'Time'));

%% 4. Convert to timetable and resample to desired interval
T = table2timetable(dataFiltered, 'RowTimes', 'Time');
T_resampled = retime(T, 'regular', 'mean', 'TimeStep', minutes(intervalMinutes));

%% 5. Validate all parameters exist
for i = 1:length(parameters)
    if ~ismember(parameters{i}, T_resampled.Properties.VariableNames)
        error('Parameter "%s" not found in dataset.', parameters{i});
    end
end

%% 6. Plot each parameter in separate figures
for i = 1:length(parameters)
    param = parameters{i};
    
    % Create new figure for each parameter
    fig = figure('Color', 'w', 'Position', [100, 100, figureWidth, figureHeight]);
    
    % Plot with specified color
    plot(T_resampled.Time, T_resampled.(param), ...
        'LineWidth', lineWidth, 'Color', lineColors{i});
    
    % Apply grid
    if gridOn
        grid on;
    else
        grid off;
    end
    
    % Set axis labels with user-defined text
    xlabel(xAxisLabels{i}, 'FontSize', axisFontSize, 'FontWeight', 'bold');
    ylabel(yAxisLabels{i}, 'FontSize', axisFontSize, 'FontWeight', 'bold');
    
    % Set title with user-defined text
    title(plotTitles{i}, 'FontSize', titleFontSize, 'FontWeight', 'bold');
    
    % Configure x-axis ticks at specific hour intervals
    tickPositions = startDate:hours(tickIntervalHours):endDate;
    xticks(tickPositions);
    
    % Set tick label format
    xtickformat(xTickFormat);
    
    % Remove the secondary date label (removes "Jan2025 - Jan 02, 2025")
    ax = gca;
    ax.XAxis.SecondaryLabel.String = '';
    
    % Rotate x-axis tick labels if specified
    if xTickRotation ~= 0
        xtickangle(xTickRotation);
    end
    
    % Set tick font size
    set(gca, 'FontSize', tickFontSize);
    box on;
    
    % Save figure
  
end

fprintf('\nAll time-series plots saved successfully!\n');
