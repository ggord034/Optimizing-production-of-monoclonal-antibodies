import os

# File locations
DIR_DATASETS = "datasets"
FILE_INITIAL_CONDITIONS = os.path.join(DIR_DATASETS, "initial_conditions.csv")

# Column names
COL_BATCH = "Batch ID"
COL_CONDITION = "Condition ID"
COL_TIME = "Time [h]"
COL_GLUCOSE = "Glucose [mM]"
COL_GLUTAMINE = "Glutamine [mM]"
COL_LACTATE = "Lactate [mM]"
COL_AMMONIA = "Ammonia [mM]"
COL_VCD = "VCD [cells/mL]"
COL_TCD = "TCD [cells/mL]"
COL_MAB = "mAb [mg/L]"
IC_FEATURES = [COL_GLUCOSE, COL_GLUTAMINE, COL_VCD]

# Economics
COST_GLUCOSE = 10.0  # [$/L per mM]
COST_GLUTAMINE = 20.0  # [$/L per mM]
COST_VCD = 0.0005  # [$/L per cells/mL]
VALUE_MAB = 1000.0  # [$/L per mg/L]
DOWNTIME = 1.0  # [d]
HOURS_PER_DAY = 24.0

# Model development
MODEL_TYPES = ["var", "poly", "knn"]
P_VALUES = [1, 2, 3]
CV_SPLITS = 3
POLY_DEGREES = [2, 2]
KNN_NEIGHBORS = [i for i in range(2, 9)]

# Forecast simulations
RANDOM_STATE = 0
PREDICTION_INTERVAL = 0.95
N_SIMULATIONS_LOBOCV = 200
N_SIMULATIONS_SCREEN = 100
N_SIMULATIONS = 200

# Number of evenly spaced levels screened for each initial condition (within the range of the training data)
SCREEN_N_LEVELS = 7
