# Columns that are not feature
NON_FEATURE_COLUMNS = [
    'customer_ID',
    'target',
    'row_number',
    'row_number_inv',
    # 'num_statements',
    'S_2'
]

# Categorical columns indicate the columns can be set as categorical
CATEGORY_COLUMNS = [
    'B_30', 'B_38', 
    'D_63', 'D_64', 'D_68', 'D_92', 
    'D_114', 'D_116', 'D_117', 'D_120', 'D_126',
    'S_binaries', 'R_binaries', 'B_binaries', 'D_binaries'
]

# Numeric columns (Requires no extra rounding)
CONTINUOUS_COLUMNS = [
    'B_1', 'B_2', 'B_3', 'B_4', 'B_5', 'B_6', 'B_7', 'B_9', 
    'B_10', 'B_11', 'B_12', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_19',
    'B_20', 'B_21', 'B_23', 'B_24', 'B_25', 'B_26', 'B_27', 'B_28',
    'B_36', 'B_37', 'B_39', 
    'B_40', 'B_42', 
    'D_42', 'D_43', 'D_45', 'D_46', 'D_47', 'D_48', 'D_49', 
    'D_50', 'D_53', 'D_55', 'D_56', 
    'D_60', 'D_61', 'D_62', 'D_65', 
    'D_71', 'D_76', 'D_73', 'D_77', 'D_88', 
    'D_104', 'D_105', 'D_106', 'D_107', 'D_108', 
    'D_110', 'D_112', 'D_113', 'D_115', 'D_118', 'D_119', 
    'D_121', 'D_122', 'D_124',
    'D_132', 'D_134', 'D_142',
    'P_2', 'P_3', 
    'R_1', 'R_6', 'R_7', 
    'R_12', 'R_14', 'R_18', 'R_26', 
    'S_3', 'S_5', 'S_7', 'S_9', 
    'S_12', 'S_16', 'S_17', 'S_19', 
    'S_22', 'S_23', 'S_24', 'S_26', 'S_27', 
]

BINARY_FEATURES = [
    "R_2", "R_4", "R_7", "R_13", "R_14", "R_15", "R_17", "R_18", "R_19", 
    "R_20", "R_21", "R_22", "R_23", "R_24", "R_25", "R_28"
]

INTEGER_FEATURES = [
    "R_1", "R_3", "R_5", "R_8", "R_9", "R_10", "R_11", "R_16", "R_26", "R_27"
]

CONTINUOUS_FEATURES = [
    "R_6", "R_12", 
]