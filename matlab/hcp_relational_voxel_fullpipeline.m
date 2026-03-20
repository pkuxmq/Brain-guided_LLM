
%% HCP RELATIONAL (VOXEL) — FULL PIPELINE
%
% This version uses the following ROI defination:
%   - Run a single SPM first-level GLM with TWO sessions (LR and RL)
%   - Compute contrast relation > match with sessrep='repl' (replicate across sessions)
%   - Select ROI inside ROI_SEARCH_MASK_PATH (e.g., MD mask) by |t| top10%
%   - Extract GLMsingle trial betas in this ROI and save final outputs

clear; clc;

%% ===================== USER SETTINGS =====================
HCP_SELECTED_DIR = './HCP_data_selected';   % <-- CHANGE
SUBJECTS = {};                          % empty -> auto-detect
SUBJECT_GROUPS = {};                    % auto-filled when SUBJECTS is empty
TASK = 'RELATIONAL';
RUNS = {'LR','RL'};
TR = 0.72;

% ---- GLMsingle stimdur settings ----
STIMDUR_STIM = 3.5;

% ---- MATCH handling as event-of-non-interest ----
MODEL_MATCH_AS_NUISANCE = false;
MATCH_STIMDUR = 2.8;
CONVOLVE_MATCH_NUISANCE = true;

% ---- Prompt regressors ----
INCLUDE_PROMPTS_AS_NUISANCE = false;
PROMPT_DUR = 2.0;

% ---- Output ----
OUTPUT_ROOT = fullfile(pwd,'HCP_RELATIONAL_VOXEL_FULLPIPELINE2');
if ~exist(OUTPUT_ROOT,'dir'); mkdir(OUTPUT_ROOT); end

% ---- GLMsingle options ----
opt = struct();
opt.wantlibrary = 1;
opt.wantglmdenoise = 1;
opt.wantfracridge = 1;
opt.wantfileoutputs = [1 1 1 1];

% ---- Block GLM (SPM multi-session) settings ----
RUN_SPM_COMBINED_GLM = true;            % set false to skip SPM combined GLM
BLOCK_GLM_INCLUDE_PROMPTS = true;
BLOCK_GLM_INCLUDE_MOTION  = true;

% ---- ROI selection (A) ----
ROI_MODE = 'top_pct';                   % 'top_pct' or 'top_n'
ROI_TOP_PCT = 10;
ROI_TOP_N   = 1000;
ROI_SEARCH_MASK_PATH = './hcp_roi/rallParcels-MD-HE197.nii';              % <-- set to MD mask NIfTI (same space as tclean); '' uses brainmask

%% Auto-detect subjects (grouped layout)
if isempty(SUBJECTS)
    SUBJECTS = {};
    SUBJECT_GROUPS = {};
    gd = dir(HCP_SELECTED_DIR);
    gd = gd([gd.isdir]);
    gd = gd(~ismember({gd.name},{'.','..'}));
    for g = 1:numel(gd)
        groupName = gd(g).name;
        sd = dir(fullfile(HCP_SELECTED_DIR, groupName));
        sd = sd([sd.isdir]);
        sd = sd(~ismember({sd.name},{'.','..'}));
        for k = 1:numel(sd)
            SUBJECTS{end+1} = sd(k).name;
            SUBJECT_GROUPS{end+1} = groupName;
        end
    end
elseif isempty(SUBJECT_GROUPS)
    SUBJECT_GROUPS = repmat({''}, size(SUBJECTS));
end
fprintf('Found %d subjects across %d groups.\n', numel(SUBJECTS), numel(unique(SUBJECT_GROUPS)));

% ---------- ROI mask reslice (once overall) ----------
if ~isempty(ROI_SEARCH_MASK_PATH) && ~isfile(ROI_SEARCH_MASK_PATH)
    if isempty(SUBJECTS)
        error('No subjects found; cannot reslice ROI mask.');
    end
    groupRef = SUBJECT_GROUPS{1};
    if isempty(groupRef)
        runDirRef = fullfile(HCP_SELECTED_DIR, SUBJECTS{1}, sprintf('tfMRI_%s_%s', TASK, RUNS{1}));
    else
        runDirRef = fullfile(HCP_SELECTED_DIR, groupRef, SUBJECTS{1}, sprintf('tfMRI_%s_%s', TASK, RUNS{1}));
    end
    tsRef = dir(fullfile(runDirRef, sprintf('tfMRI_%s_%s_hp0_clean*_tclean.nii', TASK, RUNS{1})));
    if isempty(tsRef)
        tsRef_gz = dir(fullfile(runDirRef, sprintf('tfMRI_%s_%s_hp0_clean*_tclean.nii.gz', TASK, RUNS{1})));
        file_gz = fullfile(runDirRef, tsRef_gz(1).name);
        gunzip(file_gz);
        tsRef = dir(fullfile(runDirRef, sprintf('tfMRI_%s_%s_hp0_clean*_tclean.nii', TASK, RUNS{1})));
    end
    assert(~isempty(tsRef), 'Cannot find tclean timeseries in %s', runDirRef);
    ORIGINAL_ROI_SEARCH_MASK_PATH = './hcp_roi/allParcels-MD-HE197.nii'; % <-- original MD mask NIfTI (MNI space)
    ref = sprintf('%s,1', fullfile(runDirRef, tsRef(1).name));
    flags = struct();
    flags.interp = 0;     % 0=nearest neighbor
    flags.which  = 1;
    flags.mean   = 0;
    flags.mask   = 0;
    flags.prefix = 'r';
    spm_reslice(char(ref, ORIGINAL_ROI_SEARCH_MASK_PATH), flags);
end

for s = 1:numel(SUBJECTS)
    subj = SUBJECTS{s};
    groupName = SUBJECT_GROUPS{s};
    if isempty(groupName)
        subjDir = fullfile(HCP_SELECTED_DIR, subj);
    else
        subjDir = fullfile(HCP_SELECTED_DIR, groupName, subj);
    end
    outSubj = fullfile(OUTPUT_ROOT, subj);
    if ~exist(outSubj,'dir'); mkdir(outSubj); end

    fprintf('\n==============================\nSubject %s\n==============================\n', subj);

    designs = cell(1,2);
    datas   = cell(1,2);
    extrs   = cell(1,2);
    runmeta = struct();

    condnames_full = {'REL_YES_CORR','REL_NO_CORR','REL_ERR'};
    total_counts = zeros(1,3);
    all_rel_correct_rt_ms = [];

    % ---------- build per-run GLMsingle inputs + nuisance regs for block GLM ----------
    for r=1:2
        runName = RUNS{r};
        runDir = fullfile(subjDir, sprintf('tfMRI_%s_%s', TASK, runName));
        assert(exist(runDir,'dir')==7, 'Missing runDir: %s', runDir);

        % timeseries volume
        ts = dir(fullfile(runDir, sprintf('tfMRI_%s_%s_hp0_clean*_tclean.nii', TASK, runName)));
        if isempty(ts)
            ts_gz = dir(fullfile(runDir, sprintf('tfMRI_%s_%s_hp0_clean*_tclean.nii.gz', TASK, runName)));
            file_gz = fullfile(runDir, ts_gz(1).name);
            gunzip(file_gz);
            ts = dir(fullfile(runDir, sprintf('tfMRI_%s_%s_hp0_clean*_tclean.nii', TASK, runName)));
        end
        assert(~isempty(ts), 'Cannot find tclean timeseries in %s', runDir);
        tsFile = fullfile(runDir, ts(1).name);

        maskFile = fullfile(runDir,'brainmask_fs.2.nii');
        if ~exist(maskFile,'file')
            maskFile_gz = fullfile(runDir,'brainmask_fs.2.nii.gz');
            gunzip(maskFile_gz);
            maskFile = fullfile(runDir,'brainmask_fs.2.nii');
        end
        assert(exist(maskFile,'file')==2, 'Missing brainmask_fs.2.nii in %s', runDir);

        tabHit = dir(fullfile(runDir,'*_3T_RELATIONAL_run*_TAB.txt'));
        assert(~isempty(tabHit), 'Missing TAB in %s', runDir);
        tabFile = fullfile(runDir, tabHit(1).name);

        evRelation = fullfile(runDir,'EVs','relation.txt');
        evMatch = fullfile(runDir,'EVs','match.txt');
        relEV = dlmread(evRelation);
        matchEV = dlmread(evMatch);

        movFile = fullfile(runDir,'Movement_Regressors.txt');
        motion = [];
        if exist(movFile,'file')==2
            motion = dlmread(movFile);
        end

        V = spm_vol(tsFile);
        T = numel(V);
        Vm = spm_vol(maskFile);
        brainmask = spm_read_vols(Vm) > 0.5;
        idx = find(brainmask);

        Y = zeros(numel(idx), T, 'single');
        for t=1:T
            vol = spm_read_vols(V(t));
            Y(:,t) = single(vol(idx));
        end

        if ~isempty(motion)
            if size(motion,1) ~= T
                m = min(size(motion,1), T);
                motion = motion(1:m,:);
                Y = Y(:,1:m);
                T = m;
            end
            motion = zscore(motion);
        end

        tab = readtable(tabFile,'FileType','text','Delimiter','\t','VariableNamingRule','preserve');

        % TAB->scan offset based on prompts and EV blocks
        relPrompt = tab.('RelationalPrompt.OnsetTime');
        ctrlPrompt = tab.('ControlPrompt.OnsetTime');
        relPromptS = sort(relPrompt(~isnan(relPrompt))/1000);
        ctrlPromptS = sort(ctrlPrompt(~isnan(ctrlPrompt))/1000);
        nRel = min([3, numel(relPromptS), size(relEV,1)]);
        nMatch = min([3, numel(ctrlPromptS), size(matchEV,1)]);
        if nRel==0 || nMatch==0
            error('Insufficient prompt/EV data to compute TAB2scan offset for %s %s.', subj, runName);
        end
        offRel = mean(relPromptS(1:nRel) - relEV(1:nRel,1));
        offMatch = mean(ctrlPromptS(1:nMatch) - matchEV(1:nMatch,1));
        offsetTAB2scan = mean([offRel offMatch]);
        if ~isfinite(offsetTAB2scan)
            error('Invalid TAB2scan offset (NaN/Inf) for %s %s.', subj, runName);
        end

        % REL design (time x 3)
        D = zeros(T,3);
        isRel = strcmp(string(tab.Procedure), 'RelationalPROC');
        relOn_s = (tab.('RelationalSlide.OnsetTime')(isRel)/1000) - offsetTAB2scan;
        relACC  = tab.('RelationalSlide.ACC')(isRel);
        relRT   = tab.('RelationalSlide.RT')(isRel);
        relAns  = lower(string(tab.CorrectAnswer(isRel)));

        all_rel_correct_rt_ms = [all_rel_correct_rt_ms; relRT(relACC==1)];

        condLabel = strings(numel(relOn_s),1);
        for i=1:numel(relOn_s)
            ti = round(relOn_s(i)/TR)+1;
            if ti<1 || ti>T, continue; end
            if relACC(i)==1
                if relAns(i)=="yes"; D(ti,1)=1; condLabel(i)="REL_YES_CORR";
                elseif relAns(i)=="no"; D(ti,2)=1; condLabel(i)="REL_NO_CORR";
                else; D(ti,3)=1; condLabel(i)="REL_ERR"; end
            else
                D(ti,3)=1; condLabel(i)="REL_ERR";
            end
        end
        total_counts = total_counts + sum(D,1);

        % extraregressors for GLMsingle + prompt regs for block GLM
        Xn = [];
        Pm_all = [];
        if INCLUDE_PROMPTS_AS_NUISANCE || BLOCK_GLM_INCLUDE_PROMPTS
            rp = (relPrompt(~isnan(relPrompt))/1000) - offsetTAB2scan;
            cp = (ctrlPrompt(~isnan(ctrlPrompt))/1000) - offsetTAB2scan;
            if isempty(rp) && isempty(cp)
                error('No prompt onsets found for %s %s; cannot build prompt regressors.', subj, runName);
            end
            Pm_all = zeros(T, numel(rp)+numel(cp));
            kk=0;
            for i=1:numel(rp)
                kk=kk+1;
                i0=max(1, round(rp(i)/TR)+1);
                i1=min(T, round((rp(i)+PROMPT_DUR)/TR)+1);
                Pm_all(i0:i1,kk)=1;
            end
            for i=1:numel(cp)
                kk=kk+1;
                i0=max(1, round(cp(i)/TR)+1);
                i1=min(T, round((cp(i)+PROMPT_DUR)/TR)+1);
                Pm_all(i0:i1,kk)=1;
            end
        end
        if INCLUDE_PROMPTS_AS_NUISANCE && ~isempty(Pm_all)
            Xn = [Xn Pm_all];
        end
        if ~isempty(motion)
            Xn = [Xn motion];
        end
        if MODEL_MATCH_AS_NUISANCE
            isMatch = strcmp(string(tab.Procedure), 'ControlPROC');
            mOn_s = (tab.('ControlSlide.OnsetTime')(isMatch)/1000) - offsetTAB2scan;
            x = zeros(T,1);
            durTR = max(1, round(MATCH_STIMDUR/TR));
            for i=1:numel(mOn_s)
                ti = round(mOn_s(i)/TR)+1;
                if ti<1 || ti>T, continue; end
                x(ti:min(T,ti+durTR-1)) = 1;
            end
            if CONVOLVE_MATCH_NUISANCE
                h = spm_hrf(TR);
                x = conv(x,h); x = x(1:T);
            end
            Xn = [Xn x];
        end

        designs{r} = D;
        datas{r}   = Y;
        extrs{r}   = Xn;

        % Save per-run trial table (REL only)
        meta_rel = table(string(tab.Stimulus(isRel)), string(tab.Instruction(isRel)), string(tab.CorrectAnswer(isRel)), ...
                         tab.('RelationalSlide.RESP')(isRel), relACC, relRT, relOn_s(:), condLabel, ...
            'VariableNames', {'Stimulus','Instruction','CorrectAnswer','RESP','ACC','RT_ms','Onset_s','CondLabel'});
        meta_rel = sortrows(meta_rel,'Onset_s');
        save(fullfile(outSubj, sprintf('trialmeta_%s_relational.mat',runName)), 'meta_rel','-v7.3');

        % Nuisance for SPM block GLM (prompt + motion only)
        regs_block = [];
        if BLOCK_GLM_INCLUDE_PROMPTS && ~isempty(Pm_all)
            regs_block = [regs_block Pm_all];
        end
        if BLOCK_GLM_INCLUDE_MOTION && ~isempty(motion)
            regs_block = [regs_block motion];
        end

        runmeta.(runName).runDir = runDir;
        runmeta.(runName).tsFile = tsFile;
        runmeta.(runName).maskFile = maskFile;
        runmeta.(runName).maskIdx = idx;
        runmeta.(runName).Vref = V(1);
        runmeta.(runName).brainmask = brainmask;
        runmeta.(runName).relEV = relEV;
        runmeta.(runName).matchEV = matchEV;
        runmeta.(runName).regs_block = regs_block;
    end

    % Drop empty conditions across both runs
    keep = total_counts>0;
    condnames = condnames_full(keep);
    for r=1:2
        designs{r} = designs{r}(:,keep);
    end

    stimdur_meanrt = mean(all_rel_correct_rt_ms,'omitnan')/1000;
    if isnan(stimdur_meanrt) || stimdur_meanrt<=0
        stimdur_meanrt = STIMDUR_STIM;
    end

    save(fullfile(outSubj,'subject_metadata.mat'), 'runmeta','condnames','total_counts','stimdur_meanrt','TR','STIMDUR_STIM', ...
        'MODEL_MATCH_AS_NUISANCE','ROI_MODE','ROI_TOP_PCT','ROI_TOP_N','ROI_SEARCH_MASK_PATH','-v7.3');

    %% ========== GLMsingle (cached) ==========
    opt.extraregressors = extrs;
    outStim = fullfile(outSubj,'GLMSingle_REL3_stimdurStim');
    outRT   = fullfile(outSubj,'GLMSingle_REL3_stimdurMeanRT');
    doneStim = fullfile(outStim,'TYPED_FITHRF_GLMDENOISE_RR.mat');
    doneRT   = fullfile(outRT,'TYPED_FITHRF_GLMDENOISE_RR.mat');
    if exist(doneStim,'file')~=2
        if ~exist(outStim,'dir'); mkdir(outStim); end
        GLMestimatesingletrial(designs, datas, STIMDUR_STIM, TR, outStim, opt);
    end
    if exist(doneRT,'file')~=2
        if ~exist(outRT,'dir'); mkdir(outRT); end
        GLMestimatesingletrial(designs, datas, stimdur_meanrt, TR, outRT, opt);
    end

    %% ========== SPM multi-session combined block GLM (cached) ==========
    spmCombDir = fullfile(outSubj,'SPM_blockGLM');
    tCombMat = fullfile(outSubj,'blockGLM_relationGTmatch_tstat.mat');
    if RUN_SPM_COMBINED_GLM && exist(tCombMat,'file')~=2
        % sanity: LR and RL volumes/masks must match dims
        V1 = spm_vol(runmeta.LR.tsFile); V2 = spm_vol(runmeta.RL.tsFile);
        dim1 = V1(1).dim; dim2 = V2(1).dim;
        assert(all(dim1==dim2), 'LR/RL volume dims mismatch, cannot multi-session combine.');

        if ~exist(spmCombDir,'dir'); mkdir(spmCombDir); end
        spm('defaults','fmri');
        spm_jobman('initcfg');

        matlabbatch = [];
        matlabbatch{1}.spm.stats.fmri_spec.dir = {spmCombDir};
        matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
        matlabbatch{1}.spm.stats.fmri_spec.timing.RT = TR;

        for r=1:2
            runName = RUNS{r};
            V = spm_vol(runmeta.(runName).tsFile);
            scans = cell(numel(V),1);
            for i=1:numel(V)
                scans{i} = sprintf('%s,%d', runmeta.(runName).tsFile, i);
            end

            matlabbatch{1}.spm.stats.fmri_spec.sess(r).scans = scans;

            relEV = runmeta.(runName).relEV;
            matchEV = runmeta.(runName).matchEV;

            matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(1).name = 'relation';
            matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(1).onset = relEV(:,1);
            matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(1).duration = relEV(:,2);

            matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(2).name = 'match';
            matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(2).onset = matchEV(:,1);
            matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(2).duration = matchEV(:,2);

            regs = runmeta.(runName).regs_block;
            if ~isempty(regs)
                regFile = fullfile(spmCombDir, sprintf('nuisance_regs_%s.txt', runName));
                dlmwrite(regFile, regs, 'delimiter','\t');
                matlabbatch{1}.spm.stats.fmri_spec.sess(r).multi_reg = {regFile};
            else
                matlabbatch{1}.spm.stats.fmri_spec.sess(r).multi_reg = {''};
            end

            matlabbatch{1}.spm.stats.fmri_spec.sess(r).hpf = 128;
        end

        matlabbatch{2}.spm.stats.fmri_est.spmmat = {fullfile(spmCombDir,'SPM.mat')};

        matlabbatch{3}.spm.stats.con.spmmat = {fullfile(spmCombDir,'SPM.mat')};
        matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = 'relation>match';
        matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = [1 -1];
        matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'repl'; % replicate across sessions

        spm_jobman('run', matlabbatch);

        % Extract t-stat at brainmask voxels (use LR mask idx)
        tnii = fullfile(spmCombDir,'spmT_0001.nii');
        Vt = spm_vol(tnii);
        tvol = spm_read_vols(Vt);
        tstat_comb = tvol(runmeta.LR.maskIdx);
        save(tCombMat,'tstat_comb');
    elseif exist(tCombMat,'file')==2
        tmp = load(tCombMat); tstat_comb = tmp.tstat_comb;
    else
        error('Combined SPM block GLM not available.');
    end

    %% ========== ROI selection in MD mask (or brainmask) ==========
    baseIdx = (1:numel(runmeta.LR.maskIdx))';
    if isempty(ROI_SEARCH_MASK_PATH)
        searchIdx = baseIdx;
    else
        Vm = spm_vol(ROI_SEARCH_MASK_PATH);
        % mv = spm_read_vols(Vm) > 0.5;
        mv = spm_read_vols(Vm) > 0.1;
        sel = mv(runmeta.LR.maskIdx);
        searchIdx = find(sel);
    end

    vals = abs(tstat_comb(searchIdx));
    if strcmp(ROI_MODE,'top_n')
        k = min(ROI_TOP_N, numel(vals));
    else
        k = max(1, round(numel(vals)*ROI_TOP_PCT/100));
    end
    [~,ord] = sort(vals,'descend');
    roi_combined = searchIdx(ord(1:k));

    save(fullfile(outSubj,'ROI_indices.mat'),'roi_combined','ROI_MODE','ROI_TOP_PCT','ROI_TOP_N','ROI_SEARCH_MASK_PATH');

    %% ========== FINAL extraction (ONE ROI, ALL trials LR+RL) ==========
    S_stim = load(fullfile(outStim,'TYPED_FITHRF_GLMDENOISE_RR.mat'));
    S_rt   = load(fullfile(outRT,  'TYPED_FITHRF_GLMDENOISE_RR.mat'));
    betasStim_all = S_stim.modelmd;   % [units x trials]
    betasRT_all   = S_rt.modelmd;     % [units x trials]
    all_voxels_stim = single(betasStim_all(roi_combined, :))';  % Trials x Dim
    all_voxels_rt   = single(betasRT_all(roi_combined, :))';    % Trials x Dim

    tmp = load(fullfile(outSubj,'trialmeta_LR_relational.mat')); metaLR = tmp.meta_rel;
    tmp = load(fullfile(outSubj,'trialmeta_RL_relational.mat')); metaRL = tmp.meta_rel;

    metaAll = [metaLR; metaRL];
    trialinfo = table(metaAll.Stimulus, metaAll.ACC, 'VariableNames', {'Stimulus','ACC'});

    % Sanity align (in case of mismatch due to truncation upstream)
    nTrials_glm = size(all_voxels_stim, 1);
    if height(trialinfo) ~= nTrials_glm
        m = min(height(trialinfo), nTrials_glm);
        trialinfo = trialinfo(1:m, :);
        all_voxels_stim = all_voxels_stim(1:m, :);
        all_voxels_rt   = all_voxels_rt(1:m, :);
    end

    out_all = fullfile(OUTPUT_ROOT, 'all_results');
    if ~exist(out_all,'dir'); mkdir(out_all); end
    dirStimRoot = fullfile(out_all, 'stimdurStim');
    dirRTRoot   = fullfile(out_all, 'stimdurMeanRT');
    if ~exist(dirStimRoot,'dir'); mkdir(dirStimRoot); end
    if ~exist(dirRTRoot,'dir'); mkdir(dirRTRoot); end
    if isempty(groupName)
        dirStim = dirStimRoot;
        dirRT   = dirRTRoot;
    else
        dirStim = fullfile(dirStimRoot, groupName);
        dirRT   = fullfile(dirRTRoot, groupName);
        if ~exist(dirStim,'dir'); mkdir(dirStim); end
        if ~exist(dirRT,'dir'); mkdir(dirRT); end
    end

    all_voxels = all_voxels_stim;
    save(fullfile(dirStim, [subj '.mat']), 'all_voxels');
    all_voxels = all_voxels_rt;
    save(fullfile(dirRT, [subj '.mat']), 'all_voxels');

    writetable(trialinfo, fullfile(dirStim, [subj '_trialinfo.csv']));
    writetable(trialinfo, fullfile(dirRT,   [subj '_trialinfo.csv']));

end

fprintf('\nDONE. Root: %s\n', OUTPUT_ROOT);
