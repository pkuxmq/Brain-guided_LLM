function spm_specify1stlevel_multisession(stats_dir, f_fn, multi_reg_fn, params, all_Nt)

spm('defaults','fmri');
spm_jobman('initcfg');
design_stats = struct;

num_sess = numel(f_fn);

% SETUP BATCH JOB STRUCTURE
% dir
design_stats.matlabbatch{1}.spm.stats.fmri_spec.dir = {stats_dir}; 
% timing
design_stats.matlabbatch{1}.spm.stats.fmri_spec.timing.units = params.timing_units;
design_stats.matlabbatch{1}.spm.stats.fmri_spec.timing.RT = params.timing_RT;
design_stats.matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 32; % default 16, changed according to slice time correction
design_stats.matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 16; % default 8, changed according to slice time correction
% sess
for n_sess = 1:num_sess
    scans = {};
    for i = 1:all_Nt{n_sess}
        scans{i,1} = [f_fn{n_sess} ',' num2str(i)];
    end
    design_stats.matlabbatch{1}.spm.stats.fmri_spec.sess(n_sess).scans = scans;
    num_conds = length(params.conds{n_sess});
    for i = 1:num_conds
        design_stats.matlabbatch{1}.spm.stats.fmri_spec.sess(n_sess).cond(i).name = params.conds{n_sess}(i).cond_name;
        design_stats.matlabbatch{1}.spm.stats.fmri_spec.sess(n_sess).cond(i).onset = params.conds{n_sess}(i).cond_onset;
        design_stats.matlabbatch{1}.spm.stats.fmri_spec.sess(n_sess).cond(i).duration = params.conds{n_sess}(i).cond_duration;
        design_stats.matlabbatch{1}.spm.stats.fmri_spec.sess(n_sess).cond(i).tmod = 0;
        design_stats.matlabbatch{1}.spm.stats.fmri_spec.sess(n_sess).cond(i).pmod = struct('name', {}, 'param', {}, 'poly', {});
        design_stats.matlabbatch{1}.spm.stats.fmri_spec.sess(n_sess).cond(i).orth = 1;
    end
    design_stats.matlabbatch{1}.spm.stats.fmri_spec.sess(n_sess).multi = {''};
    design_stats.matlabbatch{1}.spm.stats.fmri_spec.sess(n_sess).regress = struct('name', {}, 'val', {});
    design_stats.matlabbatch{1}.spm.stats.fmri_spec.sess(n_sess).multi_reg = multi_reg_fn(n_sess);
    design_stats.matlabbatch{1}.spm.stats.fmri_spec.sess(n_sess).hpf = 128;
end
% fact
design_stats.matlabbatch{1}.spm.stats.fmri_spec.fact = {''};
% bases
design_stats.matlabbatch{1}.spm.stats.fmri_spec.bases.hrf = struct('derivs', params.derivs); % default [0 0]
% volt
design_stats.matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
% global
design_stats.matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
% mthresh
design_stats.matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0.8000;
% mask
design_stats.matlabbatch{1}.spm.stats.fmri_spec.mask = {''}; 
% cvi
design_stats.matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';

% RUN BATCH JOB
spm_jobman('run',design_stats.matlabbatch);