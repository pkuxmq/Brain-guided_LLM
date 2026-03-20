function spm_setupTaskContrast_multi(stats_dir, params)

% SETUP BATCH JOB STRUCTURE
spm('defaults','fmri');
spm_jobman('initcfg');
contrast = struct;
% spmmat
contrast.matlabbatch{1}.spm.stats.con.spmmat = {[stats_dir filesep 'SPM.mat']};
% consess
for i = 1:numel(params)
    contrast.matlabbatch{1}.spm.stats.con.consess{i}.tcon.name = params{i}.name;
    contrast.matlabbatch{1}.spm.stats.con.consess{i}.tcon.weights = params{i}.weights;
    contrast.matlabbatch{1}.spm.stats.con.consess{i}.tcon.sessrep = 'none';
end
% delete
contrast.matlabbatch{1}.spm.stats.con.delete = 0;
% RUN BATCH JOB
spm_jobman('run',contrast.matlabbatch);
