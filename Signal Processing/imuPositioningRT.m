function [IMU_X, IMU_Y, IMU] = imuPositioningRT(IMU,i, IMU_correction, x)

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
    
    TimeStamp_prev = IMU.TimeStamp;

    TimeStamp_now = IMU.TimeStamp;
    dt = TimeStamp_now - TimeStamp_prev;
    oZ= oZ+IMU.omegaZ(i)*dt;
    Rot_mat = [cos(oZ) sin(oZ) 0; ...
               -sin(oZ) cos(oZ) 0; ...
               0 0 1 ...
              ];
    accel = Rot_mat*[IMU.aX(i) IMU.aY(i) IMU.aZ(i)]';
    speed = speed + accel*dt;
    pos = pos + speed*dt;

    IMU.IMU_X = pos(1);
    IMU.IMU_Y = pos(2);
    IMU.TimeStamp_now;
end