clc; clear all;
% Initial state estimate and covariance
x_initial = [0; 0; 0]; % Initial state estimate
P_initial = eye(3); % Initial state covariance

% Measurements (example data)
% measurements = randn(3, 10); % Example: 3 measurements for 10 time steps
measurements = [[0.0, 0.0, 0.0]
[0.5, 0.1, 0.05]
[1.0, 0.2, 0.1]
[1.5, 0.35, 0.15]
[2.0, 0.5, 0.2]
[2.5, 0.7, 0.25]
[3.0, 0.9, 0.3]
[3.5, 1.2, 0.35]
[4.0, 1.5, 0.4]
[4.5, 1.9, 0.45]
[5.0, 2.4, 0.5]
[5.5, 2.8, 0.55]
[6.0, 3.3, 0.6]
[6.5, 3.9, 0.65]
[7.0, 4.5, 0.7]
[7.5, 5.1, 0.75]
[8.0, 5.8, 0.8]
[8.5, 6.6, 0.85]
[9.0, 7.3, 0.9]
[9.5, 8.1, 0.95]
[10.0, 9.0, 1.0]
[10.5, 9.8, 1.05]
[11.0, 10.7, 1.1]
[11.5, 11.7, 1.15]
[12.0, 12.6, 1.2]
[12.5, 13.6, 1.25]
[13.0, 14.7, 1.3]
[13.5, 15.7, 1.35]
[14.0, 16.8, 1.4]
[14.5, 17.9, 1.45]
[15.0, 19.0, 1.5]
[15.5, 20.2, 1.55]
[16.0, 21.4, 1.6]
[16.5, 22.6, 1.65]
[17.0, 23.9, 1.7]
[17.5, 25.2, 1.75]
[18.0, 26.5, 1.8]
[18.5, 27.9, 1.85]
[19.0, 29.3, 1.9]
[19.5, 30.7, 1.95]
[20.0, 32.1, 2.0]]';
%%

% Parameters
v = 1; % Velocity
delta_t = 0.1; % Time step

% Run Extended Kalman Filter
x_prev = x_initial;
P_prev = P_initial;
for i = 1:size(measurements, 2)
    [x_pred, P_pred, x_upd, P_upd] = EKF_Linear(x_prev, P_prev, v, x_prev(3), delta_t, measurements(:, i));

    % Store estimated states
    x_estimated(:, i) = x_upd(1:2); % Only first two coordinates
    
    % Update previous state and covariance for next iteration
    x_prev = x_upd;
    P_prev = P_upd;
    
    % Display results
    disp(['Time Step: ', num2str(i)]);
    disp('Predicted State:');
    disp(x_pred);
    disp('Updated State:');
    disp(x_upd);
    disp('Updated Covariance:');
    disp(P_upd);
    disp('--------------------------------');
end

% Plot measurements and estimated trajectory
figure;
hold on;
plot(measurements(1,:), measurements(2,:), 'bo-', 'LineWidth', 1.5); % Plot measurements
plot(x_estimated(1,:), x_estimated(2,:), 'r.-', 'LineWidth', 1.5); % Plot estimated trajectory
xlabel('X');
ylabel('Y');
title('Measured Points and Estimated Trajectory');
legend('Measurements', 'Estimated Trajectory');
grid on;
hold off;
%%
% Parameters
vr = 1; vl = 1.2; % Velocity
delta_t = 0.1; % Time step
L = 0.16;

r = (L*(vr + vl))/(2*(vr - vl));
w = (vr - vl)/L;

% Run Extended Kalman Filter
x_prev = x_initial;
P_prev = P_initial;
for i = 1:size(measurements, 2)
    [x_pred, P_pred, x_upd, P_upd] = EKF_Nonlinear(x_prev, P_prev, r, w, x_prev(3), delta_t, measurements(:, i));

    % Store estimated states
    x_estimated(:, i) = x_upd(1:2); % Only first two coordinates
    
    % Update previous state and covariance for next iteration
    x_prev = x_upd;
    P_prev = P_upd;
    
    % Display results
    disp(['Time Step: ', num2str(i)]);
    disp('Predicted State:');
    disp(x_pred);
    disp('Updated State:');
    disp(x_upd);
    disp('Updated Covariance:');
    disp(P_upd);
    disp('--------------------------------');
end

% Plot measurements and estimated trajectory
figure;
hold on;
plot(measurements(1,:), measurements(2,:), 'bo-', 'LineWidth', 1.5); % Plot measurements
plot(x_estimated(1,:), x_estimated(2,:), 'r.-', 'LineWidth', 1.5); % Plot estimated trajectory
xlabel('X');
ylabel('Y');
title('Measured Points and Estimated Trajectory');
legend('Measurements', 'Estimated Trajectory');
grid on;
hold off;
%%

function [x_pred, P_pred, x_upd, P_upd] = EKF_Linear(x_prev, P_prev, v, theta, delta_t, measurements)
% Define process noise covariance matrix Q
    Q = diag([0.2, 0.01, 0.01]); % Adjust these values according to noise characteristics
    
    % Define measurement noise covariance matrix R
    R = diag([0.01, 0.01, 0.01]); % Adjust these values according to noise characteristicsddtheta_tf
    
    % State transition matrix A
    G = [1, 0, -v*delta_t*sin(theta);
         0, 1, v*delta_t*cos(theta);
         0, 0, 1];
     
    % Predict step
    x_pred = G * x_prev; %%% use small g here
    P_pred = G * P_prev * G' + Q;
    
    % Update step
    H = eye(3); % Measurement matrix (Identity matrix)
    y = measurements - H * x_pred;
    S = H * P_pred * H' + R;
    K = P_pred * H' * inv(S);
    
    x_upd = x_pred + K * y;
    P_upd = (eye(3) - K * H) * P_pred;
end

function [x_pred, P_pred, x_upd, P_upd] = EKF_Nonlinear(x_prev, P_prev, r, w, theta, dt, measurements)
% Define process noise covariance matrix Q
    Q = diag([0.2, 0.01, 0.01]); % Adjust these values according to noise characteristics
    
    % Define measurement noise covariance matrix R
    R = diag([0.01, 0.01, 0.01]); % Adjust these values according to noise characteristics
    
    % State transition matrix A
    % Compute elements of the Jacobian matrix A
    A = eye(3);  % Initialize A as the identity matrix
    A(1, 3) = (r * cos(w) * cos(theta) - r * sin(w) * sin(theta) - r * cos(theta)) * dt;
    A(2, 3) = (r * sin(w) * cos(theta) + r * cos(w) * sin(theta) - r * sin(theta)) * dt;

    % Compute elements of the Jacobian matrix B
    B = zeros(3, 1);
    B(1) = (-r * sin(w) * sin(theta) + r * cos(w) * cos(theta)) * dt;
    B(2) = (r * cos(w) * sin(theta) + r * sin(w) * cos(theta)) * dt;
    B(3) = dt;
     
    % Predict step
    x_pred = A * x_prev + B;
    P_pred = A * P_prev * A' + Q;
    
    % Update step
    H = eye(3); % Measurement matrix (Identity matrix)
    y = measurements - H * x_pred;
    S = H * P_pred * H' + R;
    K = P_pred * H' * inv(S);
    
    x_upd = x_pred + K * y;
    P_upd = (eye(3) - K * H) * P_pred;
end