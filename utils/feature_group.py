# Columns that are not feature
NON_FEATURE_COLUMNS = [
    'customer_ID',
    'target',
    'row_number',
    'row_number_inv',
    'S_2'
]

# Categorical columns indicate the columns can be set as categorical
CATEGORY_COLUMNS = [
    'B_30', 'B_38', 
    'D_63', 'D_92',
    'D_114', 'D_117', 'D_116', 'D_120', 'D_126'
]

MEAN_FEATURES = ['B_1', 'B_2', 'B_3', 'B_4', 'B_5', 'B_6', 'B_8', 'B_9', 
                 'B_10', 'B_11', 'B_12', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_18', 'B_19', 
                 'B_20', 'B_21', 'B_22', 'B_23', 'B_24', 'B_25', 'B_26', 'B_27', 'B_28', 'B_29', 
                 'B_32', 'B_33', 'B_36', 'B_37',  
                 'B_40', 'B_41', 'B_42', 
                 'D_39', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48', 
                 'D_50', 'D_51', 'D_52', 'D_53', 'D_54', 'D_55', 'D_58', 'D_59', 
                 'D_60', 'D_61', 'D_62', 'D_65', 'D_69', 
                 'D_70', 'D_71', 'D_72', 'D_74', 'D_75', 'D_76', 'D_77', 'D_78', 
                 'D_80', 'D_82', 'D_84', 
                 'D_91', 'D_96', 
                 'D_103', 'D_104', 'D_105', 'D_112', 'D_113', 'D_115', 'D_118', 'D_119', 
                 'D_121', 'D_122', 'D_123', 'D_124', 'D_125', 'D_128', 'D_129', 
                 'D_131', 'D_132', 'D_133', 'D_134', 'D_136', 
                 'D_140', 'D_141', 'D_142', 'D_144', 'D_145', 
                 'P_2', 'P_3', 'P_4', 
                 'R_1', 'R_2', 'R_3', 'R_6', 'R_7', 'R_8', 'R_9', 
                 'R_10', 'R_11', 'R_14', 'R_15', 'R_16', 
                 'R_20', 'R_26', 'R_27', 
                 'S_3', 'S_5', 'S_6', 'S_7', 'S_8', 'S_9', 
                 'S_11', 'S_12', 'S_13', 'S_15', 'S_16', 'S_17', 'S_19', 
                 'S_22', 'S_23', 'S_25', 'S_26', 'S_27']

MIN_FEATURES = ['B_1', 'B_2', 'B_4', 'B_5', 'B_9', 
                'B_10', 'B_11', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_19', 
                'B_20', 'B_21', 'B_24', 'B_26', 'B_28', 'B_29', 
                'B_36', 'B_42', 
                'D_42', 'D_43', 'D_45', 'D_46', 'D_47', 'D_48', 
                'D_50', 'D_51', 'D_52', 'D_53', 'D_55', 'D_56', 'D_58', 'D_59', 
                'D_60', 'D_61', 'D_62', 'D_69', 
                'D_71', 'D_74', 'D_75', 'D_77',
                'D_102', 
                'D_112', 'D_115', 'D_118', 'D_119', 
                'D_121', 'D_122', 'D_128', 
                'D_132', 'D_133', 
                'D_141', 'D_144', 
                'P_2', 'P_3', 
                'R_1', 'R_6', 'R_27', 
                'S_3', 'S_5', 'S_7', 'S_9', 
                'S_11', 'S_12', 'S_16', 'S_17', 'S_19', 
                'S_23', 'S_25', 'S_26', 'S_27']

MAX_FEATURES = ['B_2', 'B_3', 'B_4', 'B_5', 'B_6', 'B_7', 'B_8', 'B_9', 
                'B_12', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_18', 'B_19', 
                'B_21', 'B_23', 'B_24', 'B_25', 'B_26', 'B_27', 'B_29', 
                'B_36', 'B_37', 'B_40', 'B_42', 
                'D_39', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48', 'D_49', 
                'D_50', 'D_52', 'D_55', 'D_56', 'D_58', 'D_59', 
                'D_60', 'D_61', 'D_62', 'D_65', 'D_69', 
                'D_70', 'D_71', 'D_72', 'D_73', 'D_74', 'D_76', 'D_77', 'D_78', 
                'D_80', 'D_82', 'D_88',
                'D_102', 'D_105', 
                'D_110', 'D_112', 'D_115', 'D_118', 'D_119', 
                'D_121', 'D_122', 'D_123', 'D_124', 'D_125', 'D_128', 
                'D_131', 'D_132', 'D_133', 'D_134', 
                'D_141', 'D_142', 'D_144', 'D_145', 
                'P_2', 'P_3', 'P_4', 
                'R_1', 'R_3', 'R_5', 'R_6', 'R_7', 
                'R_10', 'R_11', 'R_14', 
                'R_26', 'R_27', 
                'S_3', 'S_5', 'S_7', 'S_8', 
                'S_11', 'S_12', 'S_13', 'S_15', 'S_16', 'S_17', 'S_19', 
                'S_22', 'S_23', 'S_24', 'S_25', 'S_26', 'S_27']

LAST_FEATURES = ['B_1', 'B_2', 'B_3', 'B_4', 'B_5', 'B_6', 'B_7', 'B_8', 'B_9', 
                 'B_10', 'B_11', 'B_12', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_18', 'B_19', 
                 'B_20', 'B_21', 'B_22', 'B_23', 'B_24', 'B_25', 'B_26', 'B_27', 'B_28', 'B_29', 
                 'B_30', 'B_36', 'B_37', 'B_38', 'B_39', 
                 'B_40', 'B_42', 
                 'D_39', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48', 'D_49', 
                 'D_50', 'D_51', 'D_52', 'D_53', 'D_54', 'D_55', 'D_56', 'D_58', 'D_59', 
                 'D_60', 'D_61', 'D_62', 'D_63', 'D_64', 'D_65', 'D_68', 'D_69', 
                 'D_70', 'D_71', 'D_72', 'D_73', 'D_75', 'D_76', 'D_77', 'D_78', 'D_79', 
                 'D_80', 'D_81', 'D_82', 'D_83', 'D_86', 'D_88', 
                 'D_91', 'D_96', 'D_105', 'D_106', 
                 'D_112', 'D_114', 'D_117', 'D_118', 'D_119', 
                 'D_120', 'D_121', 'D_122', 'D_124', 'D_126', 'D_127', 
                 'D_130', 'D_131', 'D_132', 'D_133', 'D_134', 'D_138', 
                 'D_140', 'D_141', 'D_142', 'D_144', 'D_145', 
                 'P_2', 'P_3', 'P_4', 
                 'R_1', 'R_2', 'R_3', 'R_4', 'R_5', 'R_6', 'R_7', 'R_8', 'R_9', 
                 'R_10', 'R_11', 'R_12', 'R_13', 'R_14', 'R_19', 
                 'R_20', 'R_26', 'R_27', 
                 'S_3', 'S_5', 'S_7', 'S_8', 'S_9', 
                 'S_11', 'S_12', 'S_13', 'S_16', 'S_17', 'S_19', 
                 'S_22', 'S_23', 'S_24', 'S_25', 'S_26', 'S_27']

FIRST_FEATURES = [
    "B_1", "B_2", "B_3", "B_4", "B_5", "B_6", "B_7", "B_8", "B_9", 
    "B_10", "B_11", "B_12", "B_13", "B_14", "B_15", "B_17", 
    "B_21", "B_23", "B_24", "B_25", "B_26", "B_27", "B_28", "B_36", "B_37", "B_38", "B_40",
    "D_42", "D_43", "D_45", "D_46", "D_47", "D_48", 
    "D_50", "D_52", "D_55", "D_58", "D_60", "D_61", "D_62", "D_69", 
    "D_71", "D_77", 
    "D_102", "D_104", "D_105", "D_115", "D_117", "D_118", "D_119", "D_121", "D_133", "D_144", 
    "P_2", "P_3",
    "R_1", "R_6", "R_27",
    "S_3", "S_5", "S_7", "S_9", "S_12", "S_16", "S_17", "S_19", 
    "S_22", "S_23", "S_24", "S_25", "S_26", "S_27"
]

RANGE_FEATURES = [
    'B_1', 'B_2', 'B_3', 'B_4', 'B_5', 'B_6', 'B_7', 'B_8', 'B_9',
    'B_10', 'B_11', 'B_12', 'B_13', 'B_14', 'B_15', 'B_17', 'B_18', 
    'B_21', 'B_23', 'B_24','B_25', 'B_26', 'B_27', 'B_28',
    'B_36', 'B_37', 'B_40',
    'D_39', 'D_43', 'D_45', 'D_46', 'D_47', 'D_48',
    'D_50', 'D_52', 'D_53', 'D_55', 'D_56', 'D_58', 
    'D_60', 'D_61', 'D_62', 'D_69', 'D_71', 'D_77',
    'D_102', 'D_104', 'D_105', 'D_112', 'D_115', 'D_118', 'D_119', 
    'D_121', 'D_128', 'D_133', 'D_144',
    'P_2', 'P_3',
    'R_1', 'R_6', 'R_27',
    'S_3', 'S_5', 'S_7', 'S_8', 'S_9',
    'S_12', 'S_16', 'S_17', 'S_19', 
    'S_22', 'S_23', 'S_24', 'S_25', 'S_26', 'S_27'
]

VELOCITY_FEATURES = [
    'B_1', 'B_2', 'B_3', 'B_4', 'B_5', 'B_6', 'B_7', 'B_8', 'B_9',
    'B_10', 'B_11', 'B_12', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_18', 'B_19',
    'B_20', 'B_21', 'B_23', 'B_24', 'B_25', 'B_26', 'B_27', 'B_28', 
    'B_33', 'B_36', 'B_37', 'B_40',
    'D_39', 'D_41', 'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48',
    'D_50', 'D_51', 'D_52', 'D_55', 'D_58', 'D_59',
    'D_60', 'D_61', 'D_62', 'D_69', 'D_70', 'D_71', 'D_74', 'D_75', 'D_80',
    'D_102', 'D_104', 'D_107',
    'D_113', 'D_115', 'D_118', 'D_119',
    'D_121', 'D_128', 'D_129', 
    'D_133', 'D_144', 
    'P_2', 'P_3', 
    'R_1', 'R_3', 'R_6', 'R_11', 'R_27',
    'S_3', 'S_5', 'S_7', 'S_8',
    'S_11', 'S_12', 'S_13', 'S_15', 'S_16', 'S_17', 'S_19',
    'S_22', 'S_23', 'S_24', 'S_25', 'S_26', 'S_27', 
]

SPEED_FEATURES = [
    'B_1', 'B_2', 'B_3', 'B_4', 'B_5', 'B_6', 'B_7', 'B_8', 'B_9',
    'B_10', 'B_11', 'B_12', 'B_13', 'B_14', 'B_15', 'B_17', 'B_18', 
    'B_21', 'B_23', 'B_24', 'B_25', 'B_26', 'B_27', 'B_28', 
    'B_36', 'B_37', 'B_40',
    'D_39', 'D_41', 'D_43', 'D_45', 'D_46', 'D_47', 'D_48',
    'D_50', 'D_52', 'D_55', 'D_58', 'D_60', 'D_61', 'D_62', 'D_69', 'D_71',
    'D_102', 'D_104', 'D_105', 'D_115', 'D_118', 'D_119', 'D_121', 'D_133', 'D_144',
    'P_2', 'P_3',
    'R_1', 'R_6', 'R_27',
    'S_3', 'S_5', 'S_7', 'S_8',
    'S_12', 'S_16', 'S_17', 'S_19',
    'S_22', 'S_23', 'S_24', 'S_25', 'S_26', 'S_27'
]

STD_FEATURES = [
    "B_12",
]


# 1 means higher default rate
HIGH_DEFAULT_IMBALANCE_BINARIES = ["R_19_last", "R_15_last", "S_20_last"]

# 0 means lower default rate
LOW_DEFAULT_IMBALANCE_BINARIES = ["S_6_last"]


# ==========================================================================================
# ARCHIVED
# ==========================================================================================
# Columns to be impute by single value
FILLNA_COLUMNS = {
    'D_87': 0  # Means column D_87 need to be fillna(0)
}

# Columns to generate another column just to see if this column has value
HAS_VALUE_COLUMNS = [
    'D_137'
]

# Binary columns indicate the columns can be round() into 1 and 0 (is float in original csv)
BINARY_COLUMNS = [
    'B_8', 
    'B_31', 'B_32', 'B_33', 
    'R_2', 'R_4',
    'R_15', 'R_19',
    'R_21', 'R_22', 'R_23', 'R_24', 'R_25', 'R_27', 'R_28',
    'S_6', 
    'S_18', 'S_20', 
    'D_54', 
    'D_86', 
    'D_93', 'D_94', 'D_96', 
    'D_103', 'D_109', 
    'D_127', 'D_128', 'D_129', 
    'D_130', 'D_135', 'D_139', 'D_140', 'D_143'
]

# ROUND COLUMNS, key = column to round, value = multiply before rounding to integer
ROUND_COLUMNS = {
    # 'B_18': 10,
    'B_22': 2,
    'B_41': 1, 
    'D_44': 8, 
    'D_51': 10/3,
    'D_70': 4, 
    'D_72': 10/3,
    'D_78': 2,
    'D_79': 2,
    'D_80': 5,
    'D_81': 1, 
    'D_82': 2,
    'D_83': 1,
    'D_84': 2,
    'D_89': 9,
    'D_91': 2,
    'D_111': 2,
    'D_123': 1,
    'D_125': 1, 
    'D_136': 4,
    'D_138': 2,
    'D_145': 11,
    'R_3': 10, 
    'R_5': 2, 
    'R_8': 1, 
    'R_9': 6, 
    'R_10': 1, 
    'R_11': 2,
    'R_13': 31,
    'R_16': 2, 
    'R_17': 35,
    'R_20': 1, 
    'S_11': 25, # +5
    'S_13': 25, 
    'S_15': 10, # +2
}

# Columns required special handling
SPECIAL_COLUMNS = [
    'B_29', 
    'D_39', 'D_41',
    'D_52', 'D_58', 'D_59',
    'D_69', 'D_74', 'D_75', 
    'D_102', 'D_131', 'D_133',
    'D_141', 'D_144', 
    'P_4',
    'S_8', 'S_25', 
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