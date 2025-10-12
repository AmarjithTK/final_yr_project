%% Time-Series Plotter for CSV (InfluxDB-style timestamps)
% Author: Amarjith TK
% Purpose: Filter and plot time-series data with flexible date and parameter selection

clc; clear; close all;

%% ================== USER DEFINED INPUTS ==================
filename = 'data.csv';                 % CSV file name
startDate = datetime('2025-01-01');    % Start date for analysis
endDate   = datetime('2025-12-31');    % End date for analysis
intervalMinutes = 15;                  % Sampling interval (minutes)
parameter = 'P';                       % Column name to plot (example: 'P', 'Q', 'V', 'I', 'KW')


%% 1. Read dataset
data = readtable(filename);

% Assuming timestamp column name could be 'timestamp', 'time', 'DateTime', etc.
% Adjust automatically if possible
timeCol = find(strcmpi(data.Properties.VariableNames, 'timestamp') | ...
               strcmpi(data.Properties.VariableNames, 'time') | ...
               strcmpi(data.Properties.VariableNames, 'datetime'), 1);

if isempty(timeCol)
    error('Timestamp column not found. Ensure a column named "timestamp" or "time" exists.');
end

% Convert to MATLAB datetime (InfluxDB timestamps: UTC ISO8601 format)
data.Time = datetime(data{:, timeCol}, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss''Z''', 'TimeZone', 'UTC');

%% 2. Filter by time range
mask = (data.Time >= startDate) & (data.Time <= endDate);
dataFiltered = data(mask, :);

if isempty(dataFiltered)
    error('No data found in the given date range.');
end

%% 3. Resample data to desired interval (optional)
% If intervalMinutes is specified, aggregate using retime
T = table2timetable(dataFiltered, 'RowTimes', 'Time');
T_resampled = retime(T, 'regular', 'mean', 'TimeStep', minutes(intervalMinutes));

%% 4. Check parameter validity
if ~ismember(parameter, T_resampled.Properties.VariableNames)
    error('Parameter "%s" not found in dataset.', parameter);
end

%% 5. Plot
figure('Color', 'w');
plot(T_resampled.Time, T_resampled.(parameter), 'LineWidth', 1.4);
grid on;
xlabel('Time (UTC)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel(sprintf('%s Value', parameter), 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('Time Series of %s (%s to %s)', parameter, ...
    datestr(startDate, 'yyyy-mm-dd'), datestr(endDate, 'yyyy-mm-dd')), ...
    'FontSize', 14, 'FontWeight', 'bold');
set(gca, 'FontSize', 11);

%% 6. Optional: Save figure
saveas(gcf, sprintf('%s_plot_%s_to_%s.png', parameter, ...
    datestr(startDate, 'yyyymmdd'), datestr(endDate, 'yyyymmdd')));
