function [x_pred, P_pred, x_upd, P_upd] = EKF_Nonlinear(x_prev, P_prev, r, w, dt, z, H, P, Q, R, I)
% Define process noise covariance matrix Q

    % x_prev(3) = z(3);



    % Q = diag([0.2, 0.01, 0.01]); % Adjust these values according to noise characteristics
    
    % Define measurement noise covariance matrix R
    % R = diag([0.01, 0.01, 0.01]); % Adjust these values according to noise characteristics
    
    % State g matrix
    g = [x_prev(1) + r*sin(x_prev(3))*cos(w*dt) + r*cos(x_prev(3))*sin(w*dt) - r*sin(x_prev(3)); 
         x_prev(2) + r*sin(w*dt)*sin(x_prev(3)) - r*cos(x_prev(3))*cos(w*dt) + r*cos(x_prev(3));
         x_prev(3) + w*dt];
    
    % State transition matrix G
    G = eye(3);  % Initialize A as the identity matrix;

    % Compute elements of the Jacobian matrix A 
    G(1, 3) = r*(cos(w*dt)*cos(x_prev(3)) - sin(x_prev(3))*sin(w*dt) - cos(x_prev(3)));
    G(2, 3) = r*(cos(w*dt)*sin(x_prev(3)) + cos(x_prev(3))*sin(w*dt) - sin(x_prev(3)));

    % Predict step
    x_pred = g; %%% use small g here
    P_pred = G * P_prev * G' + Q;
    
    % Update step
    % H = eye(3); % Measurement matrix (Identity matrix)
    S = H * P_pred * H' + R;
    K = P_pred * H'/S;
    
    x_upd = x_pred + K * (z - H * x_pred);
    P_upd = (eye(3) - K * H) * P_pred;
end