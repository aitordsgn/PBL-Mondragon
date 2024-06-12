function [IMU_X, IMU_Y, IMU_Z] = imuPositioningCamb(IMU,i, IMU_correction, x)

    persistent oZ speed pos

    if i==1
        oZ = 0;
        speed = [0 0 0]';
        pos = [0 0 0]';
    end

    if(IMU_correction==1)
        pos(1) = x(1);
        pos(2) = x(2);
        pos(3) = x(3);
    end

    dt = IMU.TimeStep(i+1)-IMU.TimeStep(i);
    oZ= oZ+IMU.omegaZ(i)*dt;
    Rot_mat = [cos(oZ) sin(oZ) 0; ...
               -sin(oZ) cos(oZ) 0; ...
               0 0 1 ...
              ];
    accel = Rot_mat*[IMU.aX(i) IMU.aY(i) IMU.aZ(i)]';
    speed = speed + accel*dt;
    pos = pos + speed*dt;

    IMU_X = pos(1);
    IMU_Y = pos(2);
    IMU_Z = pos(3);
end