clear all; close all; clc;
load('Poses/Stereo_poses.mat');
load('CameraParams.mat');
load('gTc.mat');

imageFiles = dir(fullfile('Reconstruction/', '*.bmp')); 
images = cell(1, numel(imageFiles));
for i = 1:numel(imageFiles)
    imagePath = fullfile('Reconstruction/', imageFiles(i).name);
    images{i} = imread(imagePath);
end

ptCloud = pointCloud(zeros(0, 3));
bTgi = get_T_pose(AEul, BEul, CEul,X,Y,Z);
intrinsics = cameraParams.Intrinsics;

tolerance = [1e-6,1e-6];

left = [1 2;
        2 3;
        3 4;
        4 5;
        5 6;
        8 9;
        10 11;
        11 12;
        12 13];

right = [
        %23 24;
        %24 25;
        27 28;
        28 29;
        30 31;
        %34 35;
        37 38];

top = [44 45;
       46 47];

combinations = vertcat(flipud(left),flipud(right), flipud(top));
n = length(combinations);

for j = 1:n

    disp("#################################################");
    disp("COMBINATIONS:");
    disp(combinations(j,:));
    disp("#################################################");

    TA = bTgi(:,:,combinations(j,1)) * estimated_g_Tc_dual;
    TB = bTgi(:,:,combinations(j,2)) * estimated_g_Tc_dual;
    TC = homogInv(homogInv(TA) * TB);
    module_t = norm(TC(1:3,4));
    
    img1 = images{combinations(j,1)};
    img2 = images{combinations(j,2)};
    
    % Mostrar las imágenes rectificadas
    % figure;
    % subplot(1, 2, 1);
    % imshow(img1);
    % title('Imagen Uno Orig');
    % subplot(1, 2, 2);
    % imshow(img2);
    % title('Imagen Dos Orig');
    
    undistortedImg1 = undistortImage(img1, cameraParams);
    undistortedImg2 = undistortImage(img2, cameraParams);
    
    if size(undistortedImg1, 3) == 3
        I1 = rgb2gray(undistortedImg1);
    end
    if size(undistortedImg2, 3) == 3
        I2 = rgb2gray(undistortedImg2);
    end
    
    % Detect interest points using the SURF feature detector
    % points1 = detectSURFFeatures(I1);
    % points2 = detectSURFFeatures(I2);
    points1 = detectSIFTFeatures(I1);
    points2 = detectSIFTFeatures(I2);

    % Extract feature descriptors for the detected interest points
    [features1, valid_points1] = extractFeatures(I1, points1);
    [features2, valid_points2] = extractFeatures(I2, points2);
    
    % Match feature descriptors using the ratio test
    indexPairs = matchFeatures(features1, features2, 'Unique', true, 'MatchThreshold', 10);
    
    % Retrieve the matched points
    matchedPoints1 = valid_points1(indexPairs(:, 1));
    matchedPoints2 = valid_points2(indexPairs(:, 2));
    
    % Visualizar los emparejamientos
    % figure;
    % showMatchedFeatures(undistortedImg1, undistortedImg2, matchedPoints1, matchedPoints2);
    % legend('Matched points 1','Matched points 2');
    
    % Estimate the fundamental matrix
    [E, epipolarInliers] = estimateEssentialMatrix(...
        matchedPoints1, matchedPoints2, intrinsics, Confidence = 99.99);
    
    % Find epipolar inliers
    inlierPoints1 = matchedPoints1(epipolarInliers, :);
    inlierPoints2 = matchedPoints2(epipolarInliers, :);
    
    % Display inlier matches
    % figure
    % showMatchedFeatures(I1, I2, inlierPoints1, inlierPoints2);
    % title("Epipolar Inliers");
    
    % Decompose the essential matrix into possible rotations and translations
    relPose = estrelpose(E, intrinsics, inlierPoints1, inlierPoints2);
    relPose.Translation = (relPose.Translation ./ norm(relPose.Translation)); 
    relPose.A(1:3,4) = (relPose.Translation .* module_t)'; 
    
    Transformada_Estimada = rigidtform3d(homogInv(relPose.A));
    
    T12 = homogInv(homogInv(TA) * TB); 
    Transformada_Poses = rigidtform3d(T12);
    
    % Create a stereoParameters object
    stereoParams = stereoParameters(cameraParams, cameraParams, Transformada_Estimada);
    [J1, J2, reprojectionMatrix, ~, ~, camR1, ~] = rectifyStereoImages(undistortedImg1, undistortedImg2, stereoParams, OutputView="full");
    camR_h = [camR1 [0; 0; 0;]; [0 0 0 1]];

    % Display rectified images
    % figure;
    % subplot(1, 2, 1);
    % imshow(J1);
    % title('Imagen Uno Rectificada');
    % subplot(1, 2, 2);
    % imshow(J2);
    % title('Imagen Dos Rectificada');
    
    ob1 = extraerObjeto(J1);
    ob2 = extraerObjeto(J2);
  
    disparityRange = DisparityRangeCalculation(ob1, ob2);   
    disparityMap = disparitySGM(rgb2gray(ob1),rgb2gray(ob2),"DisparityRange",disparityRange,"UniquenessThreshold",15);

    % Mostrar el mapa de disparidad
    % figure;
    % imshow(disparityMap,disparityRange)
    % title('Disparity Map')
    % colormap jet
    % colorbar
    
    % Reconstruir la nube de puntos en 3D
    points3D = reconstructScene(disparityMap, reprojectionMatrix);

    % Convertir a formato Mx3
    points3D = reshape(points3D, [], 3);
    colors = reshape(J1, [], 3);
   
    %Convertir puntos3D a coordenadas homogéneas
    numPoints = size(points3D, 1);
    points3D_homogeneous = [points3D, ones(numPoints, 1)];
    
    % Aplicar la transformación
    transformedPoints3D_homogeneous = (TA * homogInv(camR_h) * points3D_homogeneous')';
    
    % Convertir de nuevo a coordenadas cartesianas
    transformedPoints3D = transformedPoints3D_homogeneous(:, 1:3);
    
    % Filtrar colores
    [pointcloud, p_colors] = filter_color(transformedPoints3D, colors);
    
    % Crear la nube de puntos temporal con los puntos transformados
    tempPtCloud = pcdownsample(pointCloud(pointcloud, 'Color', p_colors),"random",0.5,"PreserveStructure",true);
    %tempPtCloud = pcdownsample(tempPtCloud,"nonuniformGridSample",6);
    tempPtCloud= pcdenoise(tempPtCloud);

    % Concatenar la nueva nube de puntos con la existente
    if isempty(ptCloud.Location)
        ptCloud = tempPtCloud;
    else
        icp_tform = pcregistericp(tempPtCloud,ptCloud,"Metric","planeToPlaneWithColor","MaxIterations",250, "Tolerance",tolerance);
        movingReg = pctransform(tempPtCloud, icp_tform);
        ptCloud = pcmerge(ptCloud, movingReg, 0.01);

        % figure;
        % pcshow(ptCloud);
        % axis equal
        % title(combinations(j,:)');
        % xlabel('X (mm)');
        % ylabel('Y (mm)');
        % zlabel('Z (mm)');
    end
end

ptClouddenoise= pcdenoise(ptCloud);
ptCloud = pcdownsample(ptClouddenoise,"nonuniformGridSample",7);
ptCloud= pcdenoise(ptCloud);

% Mostrar la nube de puntos final
save('Reconstructed_PointCloud_Meth2.mat', 'ptCloud');
figure;
pcshow(ptCloud);
axis equal
title('Nube de Puntos 3D Acumulada');
xlabel('X (mm)');
ylabel('Y (mm)');
zlabel('Z (mm)');

[mesh, depth, perVertexDensity] = pc2surfacemesh(ptCloud, 'poisson',6);
removeDefects(mesh,"nonmanifold-edges");
% Visualize the mesh
figure;
trisurf(mesh.Faces, mesh.Vertices(:,1), mesh.Vertices(:,2), mesh.Vertices(:,3));
title('Malla');
xlabel('X (mm)');
ylabel('Y (mm)');
zlabel('Z (mm)');

%writeSurfaceMesh(mesh,"Cubo_Mesh_2.stl")
%pcwrite(ptCloud, 'Cubo_PointCloud_2.ply')
%% FUNCTIONS
function T=get_T_pose(AEul, BEul, CEul,X,Y,Z)
    T = zeros(4,4,length(AEul));
    for i=1:length(AEul)
        R=RfromEuler(AEul(i), BEul(i), CEul(i));
        t=[X(i),Y(i),Z(i)]';
        T(:,:,i)=[R t; 0 0 0 1];
    end
end

function [R] = RfromEuler(AEul, BEul, CEul)
    R = myRotz(AEul) * myRoty(BEul) * myRotx(CEul);    
end

function Rx = myRotx(ang)
    Rx = [1 0 0; 0 cosd(ang) -sind(ang); 0 sind(ang) cosd(ang)];
end

function Ry = myRoty(ang)
    Ry = [cosd(ang) 0 sind(ang); 0 1 0; -sind(ang) 0 cosd(ang)];
end

function Rz = myRotz(ang)
    Rz = [cosd(ang) -sind(ang) 0; sind(ang) cosd(ang) 0; 0 0 1];
end

function maskedImage = extraerObjeto(img)
    imagen_gris = rgb2gray(img);

    % Binarizar la imagen usando un umbral automático
    binaryImage = imbinarize(imagen_gris);
    se = strel('disk', 5); % 'disk' crea un disco con un radio de 5 píxeles
    dilatedImage = imdilate(binaryImage, se);
    
    % Aplicar la máscara lógica a la imagen
    % Si la imagen es en color, asegúrate de aplicar la máscara a cada canal
    if size(img, 3) == 3
        maskedImage = bsxfun(@times, img, cast(dilatedImage, 'like', img));
    else
        maskedImage = I;
        maskedImage(~mask) = 0;
    end
end

function y = homogInv(M)
    R_inv = M(1:3,1:3)';
    t_inv = -R_inv * M(1:3,4);
    y = [R_inv, t_inv; 0, 0, 0, 1];
end

function [filter_pointcloud, filter_colors] = filter_color(pointcloud, colors)
    
    % Encontrar índices de puntos de color negro
    negro_indices = all(colors < 50, 2);
    % Filtrar los puntos y colores que no son negros
    puntos_filtrados = pointcloud(~negro_indices, :);
    colores_filtrados = colors(~negro_indices, :);

    % Encontrar índices de puntos de color negro
    nan_indices = all(isnan(puntos_filtrados), 2);
    % Filtrar los puntos y colores que no son negros
    filter_pointcloud = puntos_filtrados(~nan_indices, :);
    filter_colors = colores_filtrados(~nan_indices, :);
end

function disparityRange = DisparityRangeCalculation(J1, J2)

    % Detect interest points using the SURF feature detector
    points1 = detectSIFTFeatures(rgb2gray(J1));
    points2 = detectSIFTFeatures(rgb2gray(J2));
    
    % Extract feature descriptors for the detected interest points
    [features1, valid_points1] = extractFeatures(rgb2gray(J1), points1);
    [features2, valid_points2] = extractFeatures(rgb2gray(J2), points2);
    
    % Match feature descriptors using the ratio test
    indexPairs = matchFeatures(features1, features2, 'MaxRatio', 0.6, 'Unique', true);
    
    % Retrieve the matched points
    matchedPoints1 = valid_points1(indexPairs(:, 1));
    matchedPoints2 = valid_points2(indexPairs(:, 2));

    % Visualizar los emparejamientos
    % figure;
    % showMatchedFeatures(J1, J2, matchedPoints1, matchedPoints2);
    % legend('Matched points 1','Matched points 2');

    [~, inliersIndex] = estimateFundamentalMatrix(matchedPoints1, matchedPoints2, 'Method', 'RANSAC', 'NumTrials', 2000, 'DistanceThreshold', 1e-4);
    
    % Select inlier points
    inlierPoints1 = matchedPoints1(inliersIndex, :);
    inlierPoints2 = matchedPoints2(inliersIndex, :);

    % Calculate disparities for each matched point pair
    check1 = (inlierPoints1.Location(:, 1));
    check2 = (inlierPoints2.Location(:, 1));
    disparities = abs(check1 - check2);
    
    % Find the minimum and maximum disparities
    minDisparity = min(disparities);
    maxDisparity = max(disparities);

    idealCenter = (minDisparity + maxDisparity) / 2;

    minimo = idealCenter - 64;
    minAdjusted = ceil(minimo / 8) * 8;
    maxAdjusted = minAdjusted + 128;
    
    % dist = (128 - (maxDisparity - minDisparity))/2;
    % 
    % minAdjusted = minDisparity - dist;
    % maxAdjusted = maxDisparity + dist;
    % 
    % minAdjusted = ceil(minAdjusted / 8) * 8;  % Round up to the nearest multiple of 8
    % maxAdjusted = floor(maxAdjusted / 8) * 8; % Round down to the nearest multiple of 8
    % 
    % % Ensure the difference between min and max disparities is not greater than 128
    % while (maxAdjusted - minAdjusted) > 128
    %     if minAdjusted + 128 <= maxAdjusted
    %         minAdjusted = minAdjusted + 8;
    %     else
    %         maxAdjusted = maxAdjusted - 8;
    %     end
    % end

    disparityRange = [minAdjusted, maxAdjusted];
end