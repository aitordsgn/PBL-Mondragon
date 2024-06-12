function [R] = RfromEuler(AEul, BEul, CEul)
    R = myRotz(AEul) * myRoty(BEul) * myRotx(CEul);    
end

%% EULER ROTATIONS
function Rx = myRotx(ang)
    Rx = [1 0 0; 0 cosd(ang) -sind(ang); 0 sind(ang) cosd(ang)];
end

function Ry = myRoty(ang)
    Ry = [cosd(ang) 0 sind(ang); 0 1 0; -sind(ang) 0 cosd(ang)];
end

function Rz = myRotz(ang)
    Rz = [cosd(ang) -sind(ang) 0; sind(ang) cosd(ang) 0; 0 0 1];
end