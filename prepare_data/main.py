from prepare_folder import rename_folders, separate_unflagged_rgb, rename_files


rename_folders('../data/unflagged')
rename_folders('../data/flags_applied')
separate_unflagged_rgb('../data/unflagged/', '../data/unflagged_rgb/')
rename_files('../data/unflagged_rgb/')
rename_files('../data/flags_applied/')
rename_files('../data/unflagged/')