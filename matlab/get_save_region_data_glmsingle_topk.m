function get_save_region_data_glmsingle_topk(results_dir, results, roi_fn, tvalue_fn, k, suffix_setting, suffix_region, subj, save_coord, sep_voxels)
    
    Y = spm_read_vols(spm_vol(roi_fn), 1);
    assert(size(results, 1) == size(Y, 1));
    assert(size(results, 2) == size(Y, 2));
    assert(size(results, 3) == size(Y, 3));
    indx = find(Y > 0.1);
    [x, y, z] = ind2sub(size(Y), indx);
    % 3 * dim
    XYZ = [x, y, z]';

    if sep_voxels
        % separate top-k t values for syllogisms and transitive
        tvalues1 = spm_get_data(tvalue_fn{1}, XYZ);
        tvalues1 = tvalues1(:);
        tvalues2 = spm_get_data(tvalue_fn{2}, XYZ);
        tvalues2 = tvalues2(:);
        [sorted_tvalues1, idx1] = sort(tvalues1, 'descend');
        top_n1 = round(length(sorted_tvalues1) * k);
        top_idx1 = idx1(1:top_n1);
        [sorted_tvalues2, idx2] = sort(tvalues2, 'descend');
        top_n2 = round(length(sorted_tvalues2) * k);
        top_idx2 = idx2(1:top_n2);
        linear_indices = sub2ind(size(results(:,:,:,1)), XYZ(1,:), XYZ(2,:), XYZ(3,:));
        reshaped_results = reshape(results, [], size(results, 4));
        % n_questions * dim
        all_voxels = reshaped_results(linear_indices, :)';
        n_questions = size(all_voxels, 1);
        all_voxels = cat(1, all_voxels(1:n_questions/2, top_idx1), all_voxels(n_questions/2+1:n_questions, top_idx2));
    else
        % top-k t values
        if iscell(tvalue_fn)
            % if we separate Syllogisms and Transitive in the data, take the max
            tvalues1 = spm_get_data(tvalue_fn{1}, XYZ);
            tvalues1 = tvalues1(:);
            tvalues2 = spm_get_data(tvalue_fn{2}, XYZ);
            tvalues2 = tvalues2(:);
            tvalues = max(tvalues1, tvalues2);
        else
            tvalues = spm_get_data(tvalue_fn, XYZ);
            tvalues = tvalues(:);
        end
        [sorted_tvalues, idx] = sort(tvalues, 'descend');
        top_n = round(length(sorted_tvalues) * k);
        top_idx = idx(1:top_n);
    
        linear_indices = sub2ind(size(results(:,:,:,1)), XYZ(1,:), XYZ(2,:), XYZ(3,:));
        reshaped_results = reshape(results, [], size(results, 4));
        % n_questions * dim
        all_voxels = reshaped_results(linear_indices, :)';
        all_voxels = all_voxels(:, top_idx);

        region_XYZ = XYZ(:, top_idx);
    end

    voxels_dir = [results_dir filesep suffix_setting filesep 'all_extracted_beta_' suffix_region filesep subj];
    if ~exist(voxels_dir,'dir')
        mkdir(voxels_dir)
    end
    voxels_fn = [voxels_dir filesep subj '.mat'];
    save(voxels_fn, "all_voxels");
    
    save([save_coord filesep suffix_region '_region_top' num2str(k*100) '_XYZ.mat'], 'region_XYZ');

end
