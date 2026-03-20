function output = spm_standardPreproc_multirun(f_fn, s_fn, fwhm, spm_dir, all_Nt)
% Function to complete preprocessing of structural and functional data from
% a single subject with multiple runs.

% Steps include 
% 1) slice-time correction, 
% 2) motion correction,
% 3) coregistering structural image to the mean functional image,
% 4) segmenting the coregistered structural image into tissue types,
% 5) reslicing the segments to the standard MNI space, 
% 6) smoothing with Gaussian kernel.
% Makes use of spm12 batch routines. If spm12 batch parameters are not
% explicitly set, defaults are assumed. 
%
% INPUT:
% f_fn               - filenames of pre-real-time functional scans
% s_fn               - filename of T1-weighted structural scan
% fwhm               - kernel size for smoothing operations
% spm_dir            - SPM12 directory location
% all_Nt             - Nt for each functional scan
% 
% OUTPUT: 
% output            - structure with filenames and data

% Specify parameter
TR = 2; % according to the description of the fMRI data
% Declare output structure
output = struct;

% STEP 1 -- Slice time correction for each run
% interleaved slices, so slice time correction first
disp('Step 1 -- Slice time correction for functional volumes');
% run slice time correction separately
output.af_fn = cell(1, numel(f_fn));
for run_i = 1:numel(f_fn)
    f4D_spm = spm_vol(f_fn{run_i});
    nslices = f4D_spm(1).dim(3); % 32 here
    % "Slices were acquired interleaved from bottom to top with even slices acquired first"
    slice_order = [2:2:nslices 1:2:nslices];
    refslice = nslices / 2;
    spm('defaults','fmri');
    spm_jobman('initcfg');
    slice_timing = struct;
    fnms = cell(1, all_Nt{run_i});
    for i = 1:all_Nt{run_i}
        fnms{i} = [f_fn{run_i} ',' num2str(i)];
    end
    slice_timing.matlabbatch{1}.spm.temporal.st.scans = {fnms'};
    slice_timing.matlabbatch{1}.spm.temporal.st.nslices = nslices;
    slice_timing.matlabbatch{1}.spm.temporal.st.tr = TR;
    slice_timing.matlabbatch{1}.spm.temporal.st.ta = TR - (TR/nslices);
    slice_timing.matlabbatch{1}.spm.temporal.st.so = slice_order;
    slice_timing.matlabbatch{1}.spm.temporal.st.refslice = refslice;
    slice_timing.matlabbatch{1}.spm.temporal.st.prefix = 'a';
    % Run
    spm_jobman('run',slice_timing.matlabbatch);
    [d, f, e] = fileparts(f_fn{run_i});
    output.af_fn{run_i} = [d filesep 'a' f e];
end
disp('Step 1 - Done!');

% STEP 2 -- Realign (estimate and reslice) all functionals to the mean
% image of the first functional run
disp('Step 2 -- Realign all volumes');
spm('defaults','fmri');
spm_jobman('initcfg');
realign_estimate_reslice = struct;
all_scans = cell(1, numel(f_fn));
for run_i = 1:numel(f_fn)
    fnms = cell(1, all_Nt{run_i});
    for i = 1:all_Nt{run_i}
        fnms{i} = [output.af_fn{run_i} ',' num2str(i)];
    end
    all_scans{run_i} = cellstr(fnms');
end
realign_estimate_reslice.matlabbatch{1}.spm.spatial.realign.estwrite.data = all_scans';
% Eoptions
realign_estimate_reslice.matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.quality = 0.9;
realign_estimate_reslice.matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.sep = 4;
realign_estimate_reslice.matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.fwhm = 5;
realign_estimate_reslice.matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.rtm = 1;
realign_estimate_reslice.matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.interp = 2;
realign_estimate_reslice.matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.wrap = [0 0 0];
realign_estimate_reslice.matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.weight = '';
% Roptions
realign_estimate_reslice.matlabbatch{1}.spm.spatial.realign.estwrite.roptions.which = [2 1];
realign_estimate_reslice.matlabbatch{1}.spm.spatial.realign.estwrite.roptions.interp = 4;
realign_estimate_reslice.matlabbatch{1}.spm.spatial.realign.estwrite.roptions.wrap = [0 0 0];
realign_estimate_reslice.matlabbatch{1}.spm.spatial.realign.estwrite.roptions.mask = 1;
realign_estimate_reslice.matlabbatch{1}.spm.spatial.realign.estwrite.roptions.prefix = 'r';
% Run
spm_jobman('run',realign_estimate_reslice.matlabbatch);
output.raf_fn = cell(1, numel(f_fn));
output.meanf_fn = cell(1, numel(f_fn));
output.mp_fn = cell(1, numel(f_fn));
output.MP = cell(1, numel(f_fn));
for run_i = 1:numel(f_fn)
    [d, f, e] = fileparts(output.af_fn{run_i});
    output.raf_fn{run_i} = [d filesep 'r' f e];
    output.meanf_fn{run_i} = [d filesep 'mean' f e];
    output.mp_fn{run_i} = [d filesep 'rp_' f '.txt'];
    output.MP{run_i} = load(output.mp_fn{run_i});
end
disp('Step 2 - Done!');

% STEP 3 -- Coregister structural image to the mean image (estimate only)
disp('Step 3 -- Coregister structural image to the mean image');
spm('defaults','fmri');
spm_jobman('initcfg');
coreg_estimate = struct;
% Ref
%coreg_estimate.matlabbatch{1}.spm.spatial.coreg.estimate.ref = {[functional4D_fn ',1']};
coreg_estimate.matlabbatch{1}.spm.spatial.coreg.estimate.ref = output.meanf_fn(1);
% Source
coreg_estimate.matlabbatch{1}.spm.spatial.coreg.estimate.source = {s_fn};
% Eoptions
coreg_estimate.matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';
coreg_estimate.matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.sep = [4 2];
coreg_estimate.matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
coreg_estimate.matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.fwhm = [7 7];
% Run
spm_jobman('run',coreg_estimate.matlabbatch);
disp('Step 3 - Done!');

% STEP 4 -- Segmentation of coregistered structural image into GM, WM, CSF, etc
% (with implicit warping to MNI space, saving forward and inverse transformations)
disp('Step 4 -- Segmentation');
spm('defaults','fmri');
spm_jobman('initcfg');
segmentation = struct;
% Channel
segmentation.matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 0.001;
segmentation.matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 60;
segmentation.matlabbatch{1}.spm.spatial.preproc.channel.write = [0 1];
segmentation.matlabbatch{1}.spm.spatial.preproc.channel.vols = {s_fn};
% Tissue
for t = 1:6
    segmentation.matlabbatch{1}.spm.spatial.preproc.tissue(t).tpm = {[spm_dir filesep 'tpm' filesep 'TPM.nii,' num2str(t)]};
    segmentation.matlabbatch{1}.spm.spatial.preproc.tissue(t).ngaus = t-1;
    segmentation.matlabbatch{1}.spm.spatial.preproc.tissue(t).native = [1 0];
    segmentation.matlabbatch{1}.spm.spatial.preproc.tissue(t).warped = [0 0];
end
segmentation.matlabbatch{1}.spm.spatial.preproc.tissue(1).ngaus = 1;
segmentation.matlabbatch{1}.spm.spatial.preproc.tissue(6).ngaus = 2;
% Warp
segmentation.matlabbatch{1}.spm.spatial.preproc.warp.mrf = 1;
segmentation.matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 1;
segmentation.matlabbatch{1}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
segmentation.matlabbatch{1}.spm.spatial.preproc.warp.affreg = 'mni';
segmentation.matlabbatch{1}.spm.spatial.preproc.warp.fwhm = 0;
segmentation.matlabbatch{1}.spm.spatial.preproc.warp.samp = 3;
segmentation.matlabbatch{1}.spm.spatial.preproc.warp.write=[1 1];
% Run
spm_jobman('run',segmentation.matlabbatch);
% Saved filenames
[d, f, e] = fileparts(s_fn);
output.forward_transformation = [d filesep 'y_' f e];
output.inverse_transformation = [d filesep 'iy_' f e];
output.gm_fn = [d filesep 'c1' f e];
output.wm_fn = [d filesep 'c2' f e];
output.csf_fn = [d filesep 'c3' f e];
output.bone_fn = [d filesep 'c4' f e];
output.soft_fn = [d filesep 'c5' f e];
output.air_fn = [d filesep 'c6' f e];
disp('Step 4 - done!');

% STEP 5 -- Normalize image to the MNI space
disp('Step 5 -- Normalize image to the MNI space');
spm('defaults','fmri');
spm_jobman('initcfg');
normalise = struct;
output.nraf_fn = cell(1, numel(f_fn));
for run_i = 1:numel(f_fn)
    fnms = cell(1, all_Nt{run_i});
    for i = 1:all_Nt{run_i}
        fnms{i} = [output.raf_fn{run_i} ',' num2str(i)];
    end
    normalise.matlabbatch{1}.spm.spatial.normalise.write.subj.def = {output.forward_transformation};
    normalise.matlabbatch{1}.spm.spatial.normalise.write.subj.resample = cellstr(fnms');
    % WOptions
    normalise.matlabbatch{1}.spm.spatial.normalise.write.woptions.bb = [-78 -112 -70
                                                              78 76 85];
    normalise.matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = [2 2 2];
    normalise.matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = 4;
    normalise.matlabbatch{1}.spm.spatial.normalise.write.woptions.prefix = 'n';
    % Run
    spm_jobman('run',normalise.matlabbatch);
    [d, f, e] = fileparts(output.raf_fn{run_i});
    output.nraf_fn{run_i} = [d filesep 'n' f e];
end
disp('Step 5 - Done!');

% STEP 6 -- Gaussian kernel smoothing of realigned data
disp('STEP 6 -- Gaussian kernel smoothing of realigned data');
spm('defaults','fmri');
spm_jobman('initcfg');
smooth = struct;
output.snraf_fn = cell(1, numel(f_fn));
for run_i = 1:numel(f_fn)
    % Data
    fnms = cell(1, all_Nt{run_i});
    for i = 1:all_Nt{run_i}
        fnms{i} = [output.nraf_fn{run_i} ',' num2str(i)];
    end
    smooth.matlabbatch{1}.spm.spatial.smooth.data = cellstr(fnms');
    % Other
    smooth.matlabbatch{1}.spm.spatial.smooth.fwhm = [fwhm fwhm fwhm];
    smooth.matlabbatch{1}.spm.spatial.smooth.dtype = 0;
    smooth.matlabbatch{1}.spm.spatial.smooth.im = 0;
    smooth.matlabbatch{1}.spm.spatial.smooth.prefix = 's';
    % Run
    spm_jobman('run',smooth.matlabbatch);
    [d, f, e] = fileparts(output.nraf_fn{run_i});
    output.snraf_fn{run_i} = [d filesep 's' f e];
end
disp('Step 6 - done!');

