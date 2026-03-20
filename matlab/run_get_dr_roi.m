%% INITIALIZATION
spm_dir = 'D:\Matlab\R2024b\toolbox\spm12';
addpath(spm_dir);
feature('DefaultCharacterSet', 'UTF-8');

refImg = '.\data\ds003076\sub-1001\func\snrasub-1001_task-Syllogisms_run-01_bold.nii';

% Talairach space
t_coords_general = [
    -45 35 10;
    -16 1 11;
    -44 26 13;
    -49 22 16;
    -48 10 24;
    -46 15 23;
    -44 5 33;
    -37 -59 38;
    -39 -57 36;
    -24 -71 40;
    -44 12 35;
    -42 10 42;
    39 10 40;
    -42 -3 39;
    21 -69 38;
    -39 -6 50;
    -40 -6 48;
    -33 -8 53;
    -35 -5 55;
    -5 3 51;
    -44 4 45;
    20 -69 41;
    24 -64 42;
    -33 -14 49;
    -33 -14 53;
    24 -9 58;
    ];

t_coords_categorical = [
    -47 12 23;
    -39 -15 47;
    8 6 3;
    -16 7 7;
    -12 0 9;
    ];

t_coords_relational = [
    22 -7 56;
    -2 -1 53;
    -37 -56 38;
    -33 -61 38;
    24 -62 40;
    -24 -69 41;
    20 -69 41;
    -9 -67 41;
    ];

% MNI space
coords_general = tal2icbm_spm(t_coords_general);
coords_categorical = tal2icbm_spm(t_coords_categorical);
coords_relational = tal2icbm_spm(t_coords_relational);

% depend on volume size, we set a minimum radius 5
radius_general = [5 5 5 5 5 5 5 10 10 6 5 5 5 5 5 7 6 5 5 8 5 5 5 5 5 5];
radius_categorical = [7 5 7 5 5];
radius_relational = [7 5 5 5 5 5 5 5];

spm('defaults','fmri');
marsbar('on');

for i = 1:size(coords_general, 1)
    params = struct;
    params.centre = coords_general(i, :);
    params.radius = radius_general(i);
    roi = maroi_sphere(params);
    if i == 1
        roi_general = roi;
    else
        roi_general = roi_general | roi;
    end
end
roi_general = label(roi_general, 'dr_general_roi');

for i = 1:size(coords_categorical, 1)
    params = struct;
    params.centre = coords_categorical(i, :);
    params.radius = radius_categorical(i);
    roi = maroi_sphere(params);
    if i == 1
        roi_categorical = roi;
    else
        roi_categorical = roi_categorical | roi;
    end
end
roi_categorical = label(roi_categorical, 'dr_categorical_roi');

for i = 1:size(coords_relational, 1)
    params = struct;
    params.centre = coords_relational(i, :);
    params.radius = radius_relational(i);
    roi = maroi_sphere(params);
    if i == 1
        roi_relational = roi;
    else
        roi_relational = roi_relational | roi;
    end
end
roi_relational = label(roi_relational, 'dr_relational_roi');

roi_all = roi_general | roi_categorical | roi_relational;

saveroi(roi_general, '.\roi\dr_general_roi.mat');
saveroi(roi_categorical, '.\roi\dr_categorical_roi.mat');
saveroi(roi_relational, '.\roi\dr_relational_roi.mat');
saveroi(roi_all, '.\roi\dr_roi.mat');

marsbar('off');