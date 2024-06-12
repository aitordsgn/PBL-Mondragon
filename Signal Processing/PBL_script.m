%% TurtlebotData

clear;
clear oZ speed pos
clc;
close all;
% 
% filename = 'subset_2024-05-24-09-33-22.bag'; % Replace with your .BAG file
% filename = 'subset_2024-06-10-14-14-14.bag'; % Replace with your .BAG file
filename = 'subset_2024-06-10-14-40-40.bag'; % Replace with your .BAG file

[vR, vL, encoderTimeStep, IMU, beaconPos, beaconTimeStep, actual_X, actual_Y, actualTimeStep] = rosbagRead_v2(filename);

%% Turtlebot parameters
L = 0.16; % [m] length of the wheel shaft % Adjust
d = 0.065; % [m] diametre of wheel         % Adjust
r = d/2;   % [m] radius of wheel         % Adjust

%% Kalman filter

dt = mean(diff(encoderTimeStep));
Fs = 1/dt; % Frequency [Hz]

x = [actual_X(1) actual_Y(1) 0]'; % [s_x, s_y, theta]; Initial value from simulation environment

H = [1 0 0; ... % Adjust according to measured outputs
     0 1 0];

P = eye(3);

% for 1st
% best parameters
% Q = 40*eye(3); % Must be adjusted
% R = 1e3*eye(2); % Must be adjusted

% for 2nd
% best
% Q = 330*eye(3); % Must be adjusted
% R = 200*eye(2); % Must be adjusted

% for 3rd
% best
Q = 380*eye(3); % Must be adjusted
R = 10*eye(2); % Must be adjusted


I = eye(3);

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
                [x_pred, P_pred, x_upd, P_upd] = EKF_Linear(x, P, (vR(i)+vL(i))/2, dt, z, H, P, Q, R, I);
                disp('lin')
            else % Non-Linear
                L = 0.16;
                r = (L*(vR(i) + vL(i)))/(2*(vR(i) - vL(i)));
                w = (vR(i) - vL(i))/L;
                [x_pred, P_pred, x_upd, P_upd] = EKF_Nonlinear(x, P, r, w, dt, z, H, P, Q, R, I);
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
xlabel('X-Coordinates')
ylabel('Y-Coordinates')
