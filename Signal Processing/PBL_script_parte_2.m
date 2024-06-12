%% TurtlebotData

clear;
clear oZ speed pos
clc;
close all;

filename = 'subset_2024-05-24-09-33-22.bag'; % Replace with your .BAG file
% filename = 'subset_2024-06-10-14-14-14.bag'; % Replace with your .BAG file
% filename = 'subset_2024-06-10-14-40-40.bag'; % Replace with your .BAG file

[vR, vL, encoderTimeStep, IMU, beaconPos, beaconTimeStep, actual_X, actual_Y, actualTimeStep] = rosbagRead_v2(filename);

%% Turtlebot parameters
L = 0.16; % [m] length of the wheel shaft % Adjust
d = 0.065; % [m] diametre of wheel         % Adjust
r = d/2;   % [m] radius of wheel         % Adjust

%% Kalman filter

dt = mean(diff(encoderTimeStep));
Fs = 1/dt; % Frequency [Hz]

x = [actual_X(1) actual_Y(1) 0 0 0]'; % [s_x, s_y, theta]; Initial value from simulation environment

H = [1 0 0 0 0; ... % Adjust according to measured outputs
     0 1 0 0 0];

P = eye(5);

% for 1st
% best parameters
Q = 40*eye(5); % Must be adjusted
R = 1e3*eye(2); % Must be adjusted

% for 2nd
% best
% Q = 330*eye(5); % Must be adjusted
% R = 200*eye(2); % Must be adjusted

% for 3rd
% best
% Q = 380*eye(5); % Must be adjusted
% R = 10*eye(2); % Must be adjusted

I = eye(5);

est_KF = zeros(length(x), length(vR)-1); % Output vector throughout time
IMU_correction = 1; % Flag to set Feedback to IMU, DO NOT CHANGE

for t=1:length(IMU.TimeStep)-1

    % IMU's sampling rate is higher, so positioning z are
    % calculated at IMU's sampling rate. Then, when IMU's timeStep  is 
    % equal to odometry's timeStep (encoderTimeStep) is used as a 
    % correction.

    [measured_X(t), measured_Y(t)] = imuPositioning(IMU,t, IMU_correction, x);
    % [measured_X(t), measured_Y(t) measured_Z(t)] = imuPositioningCamb(IMU,t, IMU_correction, x);
    IMU_correction = 0;     % at 1 this gives no trajectory for the measured 

    s = find(IMU.TimeStep(t)<=encoderTimeStep & IMU.TimeStep(t+1)>encoderTimeStep);
    
    % theta = atan2(measured_Y(2) - measured_Y(1), measured_X(2) - measured_X(1));
    % theta = atan2(measured_Y, measured_X);
    if(~isempty(s))
        for k=1:length(s)
            i = s(k); % Odometry timeSteps that match the timeSteps of IMU.
            % slope = (measured_Y(i+1) - measured_Y(i))/(measured_X(i+1)-measured_X(i));
            % theta = atan(slope);
            % theta = atan2(measured_Y(i+1) - measured_Y(i),measured_X(i+1)-measured_X(i));
            % theta_a = rad2deg(theta)
            % theta = actual_Z(i)

            z = [measured_X(t) measured_Y(t)]'; % this is z which is data from turtlebot

            %% Type you EKF code here:
            if abs(vR(i) - vL(i)) < 0.0001 % Linear
                [x_pred, P_pred, x_upd, P_upd] = EKF_Linear2(x, P, vL(i), vR(i), dt, z, H, P, Q, R, I);
                disp('lin')
            else % Non-Linear

                [x_pred, P_pred, x_upd, P_upd] = EKF_Nonlinear2(x, P, vL(i), vR(i), dt, z, H, P, Q, R, I);
                disp('non lin')
            end 

            % Update state and covariance
            x = x_upd;
            P = P_upd;

            % Store the estimated state
            est_KF(:, i) = x;

            IMU_correction = 1; % feedback to the IMU positioning model:    % lo cambio
                                %       1: Feedback ON
                                %       0: Feedback OFF
        end
    end
end

figure
plot(actual_X,actual_Y); % Only available in simulation.
hold on;
plot(measured_X,measured_Y), 
plot(est_KF(1,:),est_KF(2,:),'--')
legend('Actual', 'Measured', 'Estimated')

%%
figure
arrayy = (est_KF(5,:));
timesteps = 1:length(est_KF);
plot(timesteps, arrayy)
figure
plot(timesteps, est_KF(4,:))
%%
% plot3(est_KF(1,:), est_KF(2,:), est_KF(5,:), 'r'); % Plot the data with circle markers and lines


function [x_pred, P_pred, x_upd, P_upd] = EKF_Linear2(x_prev, P_prev, vL, vR, delta_t, z, H, P, Q, R, I)
% Define process noise covariance matrix Q
 
    % State transition matrix G
    G = [1 0 0 0 -vL*delta_t*sin(x_prev(3));
         0 1 0 0 vL*delta_t*cos(x_prev(3));
         0 0 1 0 0;
         0 0 0 1 0; % (2*vL + 2*vR)/(2*sqrt(vL^2 + vR^2))
         0 0 0 delta_t 1];
    
    % Define small g
    g = [x_prev(1) + vL*delta_t*cos(x_prev(3));
         x_prev(2) + vL*delta_t*cos(x_prev(3)); 
         x_prev(3);  
         sqrt(vL^2 + vR^2);
         x_prev(5) + delta_t*x_prev(4)];


    % Predict step
    x_pred = g; %%% use small g here
    P_pred = G * P_prev * G' + Q;
    
    % Update step
    % H = eye(3); % Measurement matrix (Identity matrix)
    S = H * P_pred * H' + R;
    K = P_pred * H'/(S);
    
    x_upd = x_pred + K * (z - H*x_pred);
    P_upd = (eye(5) - K * H) * P_pred;
end

function [x_pred, P_pred, x_upd, P_upd] = EKF_Nonlinear2(x_prev, P_prev, vL, vR, dt, z, H, P, Q, R, I)
% Define process noise covariance matrix Q

    L = 0.16;
    r = (L*(vR + vL))/(2*(vR - vL));
    w = (vR - vL)/L;

    % State g matrix
    g = [x_prev(1) + r*sin(x_prev(3))*cos(w*dt) + r*cos(x_prev(3))*sin(w*dt) - r*sin(x_prev(3)); 
         x_prev(2) + r*sin(w*dt)*sin(x_prev(3)) - r*cos(x_prev(3))*cos(w*dt) + r*cos(x_prev(3));
         x_prev(3) + w*dt;
         sqrt(vL^2 + vR^2);
         x_prev(5) + dt*x_prev(4)];
    
    % State transition matrix G
    G = eye(5);  % Initialize A as the identity matrix;

    % Compute elements of the Jacobian matrix A 
    G(1, 3) = r*(cos(w*dt)*cos(x_prev(3)) - sin(x_prev(3))*sin(w*dt) - cos(x_prev(3)));
    G(2, 3) = r*(cos(w*dt)*sin(x_prev(3)) + cos(x_prev(3))*sin(w*dt) - sin(x_prev(3)));
    G(5, 4) = dt;

    % Predict step
    x_pred = g; %%% use small g here
    P_pred = G * P_prev * G' + Q;
    
    % Update step
    % H = eye(3); % Measurement matrix (Identity matrix)
    S = H * P_pred * H' + R;
    K = P_pred * H'/S;
    
    x_upd = x_pred + K * (z - H * x_pred);
    P_upd = (eye(5) - K * H) * P_pred;
end
