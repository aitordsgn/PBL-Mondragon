%% ROSBAG file preprocessing
function [vR, vL, encoderTimeStep, IMU, beaconPos, beaconTimeStep, actual_X, actual_Y, actualTimeStep] = rosbagRead_v2(filename)

dataBag = rosbag(filename);

wheelsDataBag = select(dataBag,'Time',[dataBag.StartTime dataBag.EndTime],'Topic','/joint_states');
wheelsStructs = readMessages(wheelsDataBag,'DataFormat','struct');

encoderTimeStep = (cellfun(@(m) double(m.Header.Stamp.Sec),wheelsStructs)*1e9 + cellfun(@(m) double(m.Header.Stamp.Nsec),wheelsStructs))/(1e9);
pR = cellfun(@(m) double(m.Position(1)),wheelsStructs);
pL = cellfun(@(m) double(m.Position(2)),wheelsStructs);

dt = encoderTimeStep(2)-encoderTimeStep(1); % Sampling time [s]

omegaR = [0; diff(pR)]/dt;
omegaL = [0; diff(pL)]/dt;

d = 0.065; % [m] diametre of wheel
r = d/2;   % [m] radius of wheel

vR = r*omegaR;  % [m/s]
vL = r*omegaL;  % [m/s]

clear wheelsStructs

% IMU data

IMUDataBag = select(dataBag,'Time',[dataBag.StartTime dataBag.EndTime],'Topic','/imu');
IMUStructs = readMessages(IMUDataBag,'DataFormat','struct');

IMUTimeStep = IMUDataBag.MessageList.Time;

IMU_X = cellfun(@(m) double(m.LinearAcceleration.X),IMUStructs);
IMU_Y = cellfun(@(m) double(m.LinearAcceleration.Y),IMUStructs);
IMU_Z = cellfun(@(m) double(m.LinearAcceleration.Z),IMUStructs);

IMU_Orientation_X = cellfun(@(m) double(m.Orientation.X),IMUStructs);
IMU_Orientation_Y = cellfun(@(m) double(m.Orientation.Y),IMUStructs);
IMU_Orientation_Z = cellfun(@(m) double(m.Orientation.Z),IMUStructs);
IMU_Orientation_W = cellfun(@(m) double(m.Orientation.W),IMUStructs);

IMU_angVel_X = cellfun(@(m) double(m.AngularVelocity.X),IMUStructs);
IMU_angVel_Y = cellfun(@(m) double(m.AngularVelocity.Y),IMUStructs);
IMU_angVel_Z = cellfun(@(m) double(m.AngularVelocity.Z),IMUStructs);

IMU.aX = IMU_X;
IMU.aY = IMU_Y;
IMU.aZ = IMU_Z;
IMU.oX = IMU_Orientation_X;
IMU.oY = IMU_Orientation_Y;
IMU.oZ = IMU_Orientation_Z;
IMU.oW = IMU_Orientation_W;
IMU.omegaX = IMU_angVel_X;
IMU.omegaY = IMU_angVel_Y;
IMU.omegaZ = IMU_angVel_Z;
IMU.TimeStep = IMUTimeStep;

% Actual coordinates
actualPosition = select(dataBag,'Time',[dataBag.StartTime dataBag.EndTime],'Topic','/gazebo/model_states');
actualPositionStructs = readMessages(actualPosition,'DataFormat','struct');

actualTimeStep = actualPosition.MessageList.Time;

actual_X = cellfun(@(m) double(m.Pose(3).Position.X),actualPositionStructs);
actual_Y = cellfun(@(m) double(m.Pose(3).Position.Y),actualPositionStructs);
actual_Z = cellfun(@(m) double(m.Pose(3).Position.Z),actualPositionStructs);

actual_Orientation_X = cellfun(@(m) double(m.Pose(3).Orientation.X),actualPositionStructs);
actual_Orientation_Y = cellfun(@(m) double(m.Pose(3).Orientation.Y),actualPositionStructs);
actual_Orientation_Z = cellfun(@(m) double(m.Pose(3).Orientation.Z),actualPositionStructs);
actual_Orientation_W = cellfun(@(m) double(m.Pose(3).Orientation.W),actualPositionStructs);

measuredTimeStep = downsample(actualTimeStep,10000);
measured_X = downsample(actual_X,10000);
measured_Y = downsample(actual_Y,10000);
% measured_Z = downsample(actual_Z,10000); % I added this

measured_X = measured_X + randn(length(measured_X),1)*0.005;
measured_Y = measured_Y + randn(length(measured_Y),1)*0.005;
% measured_Z = measured_Z + randn(length(measured_Z),1)*0.005; % I added this

beaconPos = [measured_X, measured_Y]; % cambio
beaconTimeStep = measuredTimeStep;
end