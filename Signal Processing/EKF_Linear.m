function [x_pred, P_pred, x_upd, P_upd] = EKF_Linear(x_prev, P_prev, v, delta_t, z, H, P, Q, R, I)
% Define process noise covariance matrix Q


    % Q = diag([0.2, 0.01, 0.01]); % Adjust these values according to noise characteristics
    % 
    % Define measurement noise covariance matrix R
    % R = diag([0.01, 0.01, 0.01]); % Adjust these values according to noise characteristicsddx_prev(3)_tf
    % 
    % State transition matrix G
    G = [1 0 -v*delta_t*sin(x_prev(3));
         0 1 v*delta_t*cos(x_prev(3));
         0 0 1];
    
    % Define small g
    g = [x_prev(1) + v*delta_t*cos(x_prev(3));
         x_prev(2) + v*delta_t*cos(x_prev(3)); 
         x_prev(3)];


    % Predict step
    x_pred = g; %%% use small g here
    P_pred = G * P_prev * G' + Q;
    
    % Update step
    % H = eye(3); % Measurement matrix (Identity matrix)
    S = H * P_pred * H' + R;
    K = P_pred * H'/(S);
    
    x_upd = x_pred + K * (z - H*x_pred);
    P_upd = (eye(3) - K * H) * P_pred;
end