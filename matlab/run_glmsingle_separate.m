%% MATLAB BATCH SCRIPTING: 
% Processing the fMRI dataset.
% 
% Prerequisites:
% - Matlab
% - SPM12 neuroimage processing toolbox: https://www.fil.ion.ucl.ac.uk/spm/software/spm12/
% - GLMSingle toolbox: https://github.com/cvnlab/GLMsingle
% task fMRI
% 
% Dependencies:
% - spm_standardPreproc_multirun
% - spm_specify1stlevel_multisession
% - spm_estimateModel
% - spm_setupTaskContrast_multi
% - get_save_region_data_glmsingle_topk
% 

%% INITIALIZATION
% Initialize file locations, image file names and other variables
data_dir = '.\data\ds003076';
results_dir = '.\preprocessed_data_glmsinglesep_newdrroi_topksep';
spm_dir = 'D:\Matlab\R2024b\toolbox\spm12';
addpath(spm_dir);
feature('DefaultCharacterSet', 'UTF-8');

all_subj = ['sub-1001'; 'sub-1004'; 'sub-1008'; 'sub-1014'; 'sub-1017'; 'sub-1019'; 'sub-1022'; 'sub-1027'; 'sub-1030'; 'sub-1034'];
n_subj = 10;
all_task = ['Syllogisms_run-01'; 'Syllogisms_run-02'; 'Transitive_run-01'; 'Transitive_run-02'];
n_task = 4;
all_subtype = {'2_true_affirm'; '2_false_affirm'; '2_true_negate'; '2_false_negate';
               '3_true_affirm'; '3_false_affirm'; '3_true_negate'; '3_false_negate';
               };
n_subtype = 8;

fwhm = 4;  % mm
timing_RT = 2; % according to the description of the fMRI data

deductive_reasoning_roi_fn = '.\roi\dr.nii';
language_roi_fn = '.\roi\rallParcels-language-SN220.nii';
md_roi_fn = '.\roi\rallParcels-MD-HE197.nii';

%% Run
for subj_i = 1:n_subj
    subj = all_subj(subj_i,:); %'sub-1001';
    s_fn = [data_dir filesep subj filesep 'anat' filesep subj '_T1w.nii'];
    f_fn = cell(1, n_task);
    for task_i = 1:n_task
        task = all_task(task_i,:); %'Syllogisms_run-01';
        f_fn_task = [data_dir filesep subj filesep 'func' filesep subj '_task-' task '_bold.nii'];
        f_fn{task_i} = f_fn_task;
    end

    % some fMRI data have a much longer time than the experiments, we cut 
    % off the additional scans
    % we first get all Nt for each task run
    all_Nt = cell(1, numel(f_fn));
    for task_i = 1:numel(f_fn)
        f4D_spm = spm_vol(f_fn{task_i});
        spm_size = size(f4D_spm);
        Nt = spm_size(1);

        % consider cut off
        task = all_task(task_i,:); %'Syllogisms_run-01';
        tsv_fn = [data_dir filesep subj filesep 'func' filesep subj '_task-' task '_events.tsv'];
        tsv_info = readtable(tsv_fn, 'FileType', 'text', 'Delimiter', '\t', 'TreatAsEmpty', 'n/a');

        tsv_size = size(tsv_info);
        num_all = tsv_size(1);
        % "Each run ended with the presentation of a black fixation cross for 10 s."
        time_max = tsv_info(num_all, :).onset + tsv_info(num_all, :).duration + 10;
        scan_max = floor(time_max / timing_RT);

        all_Nt{task_i} = min(scan_max, Nt);
    end
        
    %% PREPROCESSING
    disp('PREPROCESSING')
    [d, f, e] = fileparts(s_fn);
    [d1, f1, e1] = fileparts(f_fn{1});
    % Preprocess structural and functional images (if not already)
    if exist([d1 filesep 'snra' f1 e1], 'file')
        disp('...preproc already done, saving variables...')
        preproc_data = struct;
        % Structural filenames
        preproc_data.forward_transformation = [d filesep 'y_' f e];
        preproc_data.inverse_transformation = [d filesep 'iy_' f e];
        preproc_data.gm_fn = [d filesep 'c1' f e];
        preproc_data.wm_fn = [d filesep 'c2' f e];
        preproc_data.csf_fn = [d filesep 'c3' f e];
        preproc_data.bone_fn = [d filesep 'c4' f e];
        preproc_data.soft_fn = [d filesep 'c5' f e];
        preproc_data.air_fn = [d filesep 'c6' f e];
        % Functional filenames
        preproc_data.snraf_fn = cell(1, numel(f_fn));
        preproc_data.mp_fn = cell(1, numel(f_fn));
        for run_i = 1:numel(f_fn)
            [d1, f1, e1] = fileparts(f_fn{run_i});
            preproc_data.snraf_fn{run_i} = [d1 filesep 'snra' f1 e1];
            preproc_data.mp_fn{run_i} = [d1 filesep 'rp_a' f1 '.txt'];
        end
    else
        disp('...running preprocessing batch jobs...')
        preproc_data = spm_standardPreproc_multirun(f_fn, s_fn, fwhm, spm_dir, all_Nt);
    end
    % Check coregistration and segmentation
    spm_check_registration([preproc_data.snraf_fn{1} ',1'])
    disp('Preprocessing done!')
    
    %% ESTIMATE RESPONSES WITH GLMSINGLE
    % first set design matrx and get the data
    % since syllogisms and transitive tasks usually have different response
    % time, while GLMsingle assumes a fixed duration, process seperately
    design_matrices_syll = cell(1, n_task/2);
    design_matrices_tran = cell(1, n_task/2);
    data_all_syll = cell(1, n_task/2);
    data_all_tran = cell(1, n_task/2);
    avg_response_time_syll = [];
    avg_response_time_tran = [];

    tsv_info_control = cell(1, n_task);
    tsv_info_trial = cell(1, n_task);
    tsv_info_subtype = cell(1, n_task);
    % we want to distinguish correct and incorrect answer, so first
    % calculate which conditions we will have
    num_per_cond_syll = zeros(n_subtype * 2);
    num_per_cond_tran = zeros(n_subtype * 2);
    for task_i = 1:n_task
        task = all_task(task_i,:); %'Syllogisms_run-01';
        tsv_fn = [data_dir filesep subj filesep 'func' filesep subj '_task-' task '_events.tsv'];
        tsv_info = readtable(tsv_fn, 'FileType', 'text', 'Delimiter', '\t', 'TreatAsEmpty', 'n/a');
        tsv_info_control{task_i} = tsv_info(strcmp(tsv_info.trial_type, 'control'), :);
        tsv_info_trial{task_i} = tsv_info(~strcmp(tsv_info.trial_type, 'control'), :);
        % trial_type has errors, use stim_file to identify subtype
        subtypes = tsv_info(~strcmp(tsv_info.trial_type, 'control'), :).stim_file;
        for t = 1 : length(subtypes)
            name = strsplit(subtypes{t}, '.');
            name = strsplit(name{1}, '_');
            subtype = [name{2} '_' name{3} '_' name{4}(1:6)];
            subtypes{t} = subtype;
        end
        tsv_info_subtype{task_i} = subtypes;

        num_trials = length(tsv_info_trial{task_i}.onset);
        for trial_index = 1 : num_trials
            subtype = tsv_info_subtype{task_i}{trial_index};
            subtype_ind = 1;
            for ind = 1 : n_subtype
                if strcmp(all_subtype(ind), subtype)
                    subtype_ind = ind;
                    break
                end
                if ind == n_subtype
                    disp('Error! Not find subtype index');
                end
            end
            if tsv_info_trial{task_i}.accuracy(trial_index) == 1
                if task_i <= 2
                    num_per_cond_syll(subtype_ind) = num_per_cond_syll(subtype_ind) + 1;
                else
                    num_per_cond_tran(subtype_ind) = num_per_cond_tran(subtype_ind) + 1;
                end
            else
                if task_i <= 2
                    num_per_cond_syll(subtype_ind+n_subtype) = num_per_cond_syll(subtype_ind+n_subtype) + 1;
                else
                    num_per_cond_tran(subtype_ind+n_subtype) = num_per_cond_tran(subtype_ind+n_subtype) + 1;
                end
            end
        end
    end
    non_conds_syll = [];
    non_conds_tran = [];
    for ii = 1:n_subtype*2
        if num_per_cond_syll(ii) == 0
            non_conds_syll = [non_conds_syll; ii];
        end
        if num_per_cond_tran(ii) == 0
            non_conds_tran = [non_conds_tran; ii];
        end
    end

    for task_i = 1:n_task
        % setup design matrix
        if task_i <= 2
            non_conds = non_conds_syll;
        else
            non_conds = non_conds_tran;
        end
        design_m = zeros(all_Nt{task_i}, n_subtype*2-length(non_conds));
        num_trials = length(tsv_info_trial{task_i}.onset);
        for trial_index = 1 : num_trials
            onset = tsv_info_trial{task_i}.onset(trial_index) + 6;
            onset_ind = floor(onset / timing_RT) + 1;
            subtype = tsv_info_subtype{task_i}{trial_index};
            subtype_ind = 1;
            for ind = 1 : n_subtype
                if strcmp(all_subtype(ind), subtype)
                    subtype_ind = ind;
                    break
                end
            end
            if tsv_info_trial{task_i}.accuracy(trial_index) == 0
                subtype_ind = subtype_ind + n_subtype;
            end
            % skip non_cond settings
            cnt = 0;
            for ii = 1:length(non_conds)
                if subtype_ind > non_conds(ii)
                    cnt = cnt + 1;
                else
                    break
                end
            end
            design_m(onset_ind, subtype_ind - cnt) = 1;
        end
        if task_i <= 2
            design_matrices_syll{task_i} = design_m;
        else
            design_matrices_tran{task_i-2} = design_m;
        end
        % get data
        file = preproc_data.snraf_fn{task_i};
        V = spm_vol(file);
        [X, Y, Z] = size(spm_read_vols(V(1)));
        data = zeros(X, Y, Z, all_Nt{task_i});
        for i = 1 : all_Nt{task_i}
            volume_file = [file ',' num2str(i)];
            V_i = spm_vol(volume_file);
            data(:, :, :, i) = spm_read_vols(V_i);
        end
        response_time = [];
        for ii = 1:length(tsv_info_trial{task_i}.response_time)
            if tsv_info_trial{task_i}.accuracy(ii) == 1
                response_time = [response_time tsv_info_trial{task_i}.response_time(ii)];
            end
        end
        if task_i <= 2
            data_all_syll{task_i} = data;
            avg_response_time_syll = [avg_response_time_syll; mean(response_time)];
        else
            data_all_tran{task_i-2} = data;
            avg_response_time_tran = [avg_response_time_tran; mean(response_time)];
        end
    end
    % average response time
    avg_rt_syll = mean(avg_response_time_syll);
    avg_rt_tran = mean(avg_response_time_tran);

    % syllogisms
    outputdir = ['GLMestimatesingletrialoutputs_sep' filesep subj filesep 'Syllogisms'];
    % call GLMsingle
    if exist([outputdir filesep 'TYPED_FITHRF_GLMDENOISE_RR.mat'], 'file')
        results = load([outputdir filesep 'TYPED_FITHRF_GLMDENOISE_RR.mat']);
    else
        [results,resultsdesign] = GLMestimatesingletrial(design_matrices_syll,data_all_syll,avg_rt_syll,timing_RT,outputdir);
        results = results{4};
    end
    betas_syll = results.modelmd;

    % transitive
    outputdir = ['GLMestimatesingletrialoutputs_sep' filesep subj filesep 'Transitive'];
    % call GLMsingle
    if exist([outputdir filesep 'TYPED_FITHRF_GLMDENOISE_RR.mat'], 'file')
        results = load([outputdir filesep 'TYPED_FITHRF_GLMDENOISE_RR.mat']);
    else
        [results,resultsdesign] = GLMestimatesingletrial(design_matrices_tran,data_all_tran,avg_rt_tran,timing_RT,outputdir);
        results = results{4};
    end
    betas_tran = results.modelmd;

    % X*Y*Z*q_ind
    betas = cat(4, betas_syll, betas_tran);

    %% GET VOXELS WITH CONSTRAST
    % we consider top k% most responsive voxels, so we first identify
    % related voxels for correct answers by contrast with the premises
    suffix_tc = 'trial_correct-premises_sep';
    sess_params = struct;
    sess_params.timing_units = 'secs';
    sess_params.timing_RT = timing_RT;
    sess_params.derivs = [0 0];
    templatestruct = struct('cond_name', [], 'cond_onset', [], 'cond_duration', []);
    sess_params.conds = cell(1, n_task);
    for task_i = 1:n_task
        sess_params.conds{task_i} = repmat(templatestruct, 3, 1);
        % trial condition
        % first filter incorrect questions
        onset_correct = [];
        response_correct = [];
        for ii = 1:length(tsv_info_trial{task_i}.onset)
            if tsv_info_trial{task_i}.accuracy(ii) == 1
                onset_correct = [onset_correct; tsv_info_trial{task_i}.onset(ii) + 6];
                response_correct = [response_correct; tsv_info_trial{task_i}.response_time(ii)];
            end
        end
        sess_params.conds{task_i}(1) = struct('cond_name', [], 'cond_onset', [], 'cond_duration', []);
        sess_params.conds{task_i}(1).cond_name = 'trial';
        sess_params.conds{task_i}(1).cond_onset = onset_correct; 
        sess_params.conds{task_i}(1).cond_duration = response_correct; 
        % premise condition
        sess_params.conds{task_i}(2) = struct('cond_name', [], 'cond_onset', [], 'cond_duration', []);
        sess_params.conds{task_i}(2).cond_name = 'premises';
        sess_params.conds{task_i}(2).cond_onset = tsv_info_trial{task_i}.onset;
        sess_params.conds{task_i}(2).cond_duration = 6;
        % control condition
        sess_params.conds{task_i}(3) = struct('cond_name', [], 'cond_onset', [], 'cond_duration', []);
        sess_params.conds{task_i}(3).cond_name = 'control';
        sess_params.conds{task_i}(3).cond_onset = tsv_info_control{task_i}.onset;
        sess_params.conds{task_i}(3).cond_duration = tsv_info_control{task_i}.duration;
    end

    % Call script to set up design
    stats_dir_tc = [data_dir filesep subj filesep suffix_tc];
    if ~exist(stats_dir_tc,'dir')
        mkdir(stats_dir_tc)
    end

    tvalue_fn = cell(1, 2);
    tvalue_fn{1} = [stats_dir_tc filesep 'spmT_' sprintf('%04d', 1) '.nii'];
    tvalue_fn{2} = [stats_dir_tc filesep 'spmT_' sprintf('%04d', 2) '.nii'];
    if ~exist(tvalue_fn{1}, 'file')
        spm_specify1stlevel_multisession(stats_dir_tc, preproc_data.snraf_fn, preproc_data.mp_fn, sess_params, all_Nt)
        % Display/explore design matrix 
        load([stats_dir_tc filesep 'SPM.mat']);
    
        % ESTIMATE MODEL
        spm_estimateModel(stats_dir_tc)
    
        % SETUP TASK CONTRAST
        [Ntt, Nregr] = size(SPM.xX.X);
        contrast_params_all = cell(1, 2);

        % Syllogisms
        contrast_params = struct;
        % order: trial, premise, control, and 6 movement parameters
        contrast_params.weights = zeros(1, Nregr); 
        % the trial condition
        contrast_params.weights(1) = 0.5;
        contrast_params.weights(10) = 0.5;
        % the premise condition
        contrast_params.weights(2) = -0.5;
        contrast_params.weights(11) = -0.5;
        contrast_params.name = 'Syllogism-Trial_correct-Premises';
        contrast_params_all{1} = contrast_params;

        % Transitive
        contrast_params = struct;
        contrast_params.weights = zeros(1, Nregr); 
        % the trial condition
        contrast_params.weights(19) = 0.5;
        contrast_params.weights(28) = 0.5;
        % the premise condition
        contrast_params.weights(20) = -0.5;
        contrast_params.weights(29) = -0.5;
        contrast_params.name = 'Transitive-Trial_correct-Premises';
        contrast_params_all{2} = contrast_params;

        spm_setupTaskContrast_multi(stats_dir_tc, contrast_params_all);
    end

    % top-10%
    k = 0.1;
    suffix = 'top-10%';
    XYZ_dir = ['region_coord' filesep subj];
    if ~exist(XYZ_dir,'dir')
        mkdir(XYZ_dir)
    end
    % Extract deductive reasoning regions
    get_save_region_data_glmsingle_topk(results_dir, betas, deductive_reasoning_roi_fn, tvalue_fn, k, suffix, 'deductive_reasoning', subj, XYZ_dir, 0);
    % Extract language regions
    get_save_region_data_glmsingle_topk(results_dir, betas, language_roi_fn, tvalue_fn, k, suffix, 'language', subj, XYZ_dir, 0);
    % Extract MD regions
    get_save_region_data_glmsingle_topk(results_dir, betas, md_roi_fn, tvalue_fn, k, suffix, 'md', subj, XYZ_dir, 0);

end