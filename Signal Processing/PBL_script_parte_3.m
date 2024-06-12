clc; clear all;
%%
% Create a subsrciber
imuSub = rossubscriber("/imu","DataFormat","struct");
jointSub = rossubscriber("/joint_states","DataFormat","struct");
% modelSub = rossubscriber("/model_states","DataFormat","struct");

% initialize timestep
encoderTimeStep = zeros(2,1);

% Turtlebot parametersros
L = 0.16; % [m] length of the wheel shaft % Adjust
d = 0.065; % [m] diametre of wheel         % Adjust
r = d/2;   % [m] radius of wheel         % Adjust
i = 1;



% read message from subscriber
% imudata = receive(imuSub,10);
% jointdata = receive(jointSub,10);
% modeldata = receive(modelSub,10);
oZ = 0;
speed = [0 0 0]';
pos = [0 0 0]';
pause(1);

x = [0 0 0]';
%%
% est_KF = zeros(length(x), length(vR)-1); % Output vector throughout time
%
i=1;
loopVal = 10000;
% Initialize new arrays to store non-zero values
nonZeroTimeStampsI = [];
nonZeroTimeStampsE = [];

pause(1);

pR = [];
pL = [];

IMU(loopVal) = struct('aX', [], 'aY', [], 'aZ', [], ...
                            'oX', [], 'oY', [], 'oZ', [], 'oW', [], ...
                            'omegaX', [], 'omegaY', [], 'omegaZ', [], ...
                            'TimeStamp', []);
timeStampsI = [];
timeStampsE = [];
aX = [];

tic
while (i<loopVal)
%parsing message from IMU
IMU_X = imuSub.LatestMessage.LinearAcceleration.X;
IMU_Y = imuSub.LatestMessage.LinearAcceleration.Y;
IMU_Z = imuSub.LatestMessage.LinearAcceleration.Z;

IMU_Orientation_X = imuSub.LatestMessage.Orientation.X;
IMU_Orientation_Y = imuSub.LatestMessage.Orientation.Y;
IMU_Orientation_Z = imuSub.LatestMessage.Orientation.Z;
IMU_Orientation_W = imuSub.LatestMessage.Orientation.W;

IMU_angVel_X = imuSub.LatestMessage.AngularVelocity.X;
IMU_angVel_Y = imuSub.LatestMessage.AngularVelocity.Y;
IMU_angVel_Z = imuSub.LatestMessage.AngularVelocity.Z;


% imuSub.LatestMessage.Header.Seq

timeStampsI = [timeStampsI imuSub.LatestMessage.Header.Stamp.Nsec/1e7];
timeStampsE = [timeStampsE jointSub.LatestMessage.Header.Stamp.Nsec/1e7];

diff_I = diff(timeStampsI);
diff_E = diff(timeStampsE);

% Find non-zero positions
non_zero_positions = find(diff_I ~= 0 & diff_E ~= 0);

% Extract values from these positions
values_diff_I = diff_I(non_zero_positions);
values_diff_E = diff_E(non_zero_positions);

% Calculate the average
average_values = (values_diff_I + values_diff_E) / 2;

% updateTimestamps(timeStampsI, timeStampsE);

dt = imuSub.LatestMessage.Header.Stamp.Nsec/1e9;
aX = [aX imuSub.LatestMessage.LinearAcceleration.X];

IMU(i).aX = IMU_X;
IMU(i).aY = IMU_Y;
IMU(i).aZ = IMU_Z;
IMU(i).oX = IMU_Orientation_X;
IMU(i).oY = IMU_Orientation_Y;
IMU(i).oZ = IMU_Orientation_Z;
IMU(i).oW = IMU_Orientation_W;
IMU(i).omegaX = IMU_angVel_X;
IMU(i).omegaY = IMU_angVel_Y;
IMU(i).omegaZ = IMU_angVel_Z;
IMU(i).TimeStamp = timeStampsI(end);



% Parsing message from a Joint states
pR = [pR jointSub.LatestMessage.Position(1,1)];
pL = [pL jointSub.LatestMessage.Position(2,1)];

IMU_correction = 1; % Flag to set Feedback to IMU, DO NOT CHANGE

% Update time step
i = i+1;
end
toc

dt = average_values;
dt = double(dt);
IMU = IMU(non_zero_positions);
pRR = pR(non_zero_positions);
pLL = pL(non_zero_positions);
% 
omegaR = pRR./dt;
omegaL = pLL./dt;
% 
vR = r*omegaR;  % [m/s]
vL = r*omegaL;  % [m/s]

% Initial value from simulation environment
x = [0 0 0];

H = [1 0 0; ... % Adjust according to measured outputs
     0 1 0];

P = eye(3);

% for 3rd
% best
Q = 330*eye(3); % Must be adjusted
R = 200*eye(2); % Must be adjusted

I = eye(3);

% 
IMU_correction = 1; % Flag to set Feedback to IMU, DO NOT CHANGE
% 
for t = 1:length(non_zero_positions)
    [measured_X(t), measured_Y(t)] = imuPositioningReal(IMU(t),t, IMU_correction, x,dt(t));

    z = [measured_X(t), measured_Y(t)]';

    if abs(vR(t) - vL(t)) < 0.0001 % Linear
                [x_pred, P_pred, x_upd, P_upd] = EKF_Linear(x, P, (vR(t)+vL(t))/2, dt(t), z, H, P, Q, R, I);
                disp('lin')
            else % Non-Linear
                L = 0.16;
                r = (L*(vR(t) + vL(t)))/(2*(vR(t) - vL(t)));
                w = (vR(t) - vL(t))/L;
                [x_pred, P_pred, x_upd, P_upd] = EKF_Nonlinear(x, P, r, w, dt(t), z, H, P, Q, R, I);
                disp('non lin')
            end 

            % Update state and covariance
            x = x_upd;
            P = P_upd;

            % Store the estimated state
            est_KF(:, t) = x;

            IMU_correction = 1; % feedback to the IMU positioning model:    % lo cambio
                                %       1: Feedback ON
                                %       0: Feedback OFF
end

% Probe plots
figure;
plot(diff(aX))
figure;
stem(diff(timeStampsI))
figure;
stem(diff(timeStampsE))

figure

plot(measured_X,measured_Y), 
hold on;
plot(est_KF(1,:),est_KF(2,:),'--')
legend('Actual', 'Measured', 'Estimated')


function [IMU_X, IMU_Y] = imuPositioningReal(IMU,i, IMU_correction, x,dt)

    persistent oZ speed pos

    if i==1
        oZ = 0;
        speed = [0 0 0]';
        pos = [0 0 0]';
    end

    if(IMU_correction==1)
        pos(1) = x(1);
        pos(2) = x(2);
    end

    disp(dt);
    a=1;
    oZ = oZ + IMU.omegaZ * dt;
    Rot_mat = [cos(oZ) sin(oZ) 0; ...
               -sin(oZ) cos(oZ) 0; ...
               0 0 1 ...
              ];
    accel = Rot_mat*[IMU.aX IMU.aY IMU.aZ]';
    speed = speed + accel*dt;
    pos = pos + speed*dt;

    IMU_X = pos(1);
    IMU_Y = pos(2);
end
