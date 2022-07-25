# ==========================================================================
# GBDT
# ==========================================================================
# Score = 0.7942, 469 features, last | min | max | avg
lgbm_hyperparams = {
    'reg_alpha': 0.09816728852981829, 
    'reg_lambda': 19.26243620892247, 
    'learning_rate': 0.017545177936201698, 
    'n_estimators': 1787, 
    'colsample_bytree': 0.19889648708686236, 
    'subsample': 0.7440378528722045, 
    'subsample_freq': 2, 
    'min_child_samples': 2118, 
    'scale_pos_weight': 1.3534535194277337, 
    'max_bins': 552, 
    'num_leaves': 123
}

# Score = 0.7942, 469 features + first
{
    'reg_alpha': 0.09920863649068341, 
    'reg_lambda': 20.12297223356303, 
    'learning_rate': 0.02793692417382347, 
    'n_estimators': 1736, 
    'colsample_bytree': 0.19068604246039428, 
    'subsample': 0.7704667015861436, 
    'subsample_freq': 4, 
    'min_child_samples': 2210, 
    'scale_pos_weight': 1.679152154421724, 
    'max_bins': 583,
    'num_leaves': 92
}

# Score = 0.7944, 
{
    'reg_alpha': 0.0017758691158297811, 
    'reg_lambda': 14.625894238087323, 
    'learning_rate': 0.017217692806146635, 
    'n_estimators': 1752, 
    'colsample_bytree': 0.17228213947525065, 
    'subsample': 0.7190986455613982, 
    'subsample_freq': 2, 
    'min_child_samples': 2110, 
    'scale_pos_weight': 1.359583733943126, 
    'max_bins': 547, 
    'num_leaves': 97
}

# Score = 0.7945, 
{
    'reg_alpha': 0.002180187257614318, 
    'reg_lambda': 12.302801030330981, 
    'learning_rate': 0.016749009699215868, 
    'n_estimators': 1988, 
    'colsample_bytree': 0.169751060866367, 
    'subsample': 0.7259738160555449, 
    'subsample_freq': 3, 
    'min_child_samples': 2198, 
    'scale_pos_weight': 1.538035475965108, 
    'max_bins': 544, 
    'num_leaves': 90
}

# Score (non CV) = 0.7946, 
{
    'reg_alpha': 0.02378710792245604, 
    'reg_lambda': 18.221287086859476, 
    'learning_rate': 0.022154753744976614, 
    'n_estimators': 2029, 
    'colsample_bytree': 0.3258846812535845, 
    'subsample': 0.6895581472815298, 
    'subsample_freq': 3, 
    'min_child_samples': 2449, 
    'scale_pos_weight': 1.4323212215746224, 
    'max_bins': 506, 
    'num_leaves': 130
}

# ==========================================================================
# Dart
# ==========================================================================
# Score = 0.7944
{
    'reg_alpha': 0.0001444366874151164, 
    'reg_lambda': 27.853926584276863, 
    'learning_rate': 0.039731503659292934, 
    'n_estimators': 2418, 
    'colsample_bytree': 0.3321195918336475, 
    'subsample': 0.8498601413055697, 
    'subsample_freq': 7, 
    'min_child_samples': 2498, 
    'scale_pos_weight': 1.9675418710851762, 
    'max_bins': 498, 
    'num_leaves': 149
}
