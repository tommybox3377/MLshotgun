from sklearn.preprocessing import QuantileTransformer, PowerTransformer, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer

# Tillio vars to send message when program is complete
twilio_ID = "ABC123"
twilio_secret_ID = "ABC123"
twilio_to = "+123456789"
twilio_from = "+123456789"

# Shoot.py vars for
workers = 2  # number of workers to be used for the grid search
folds = 4  # number of cross-validation folds used in grid search
verbose = 2  # amount of text to be printed during runtime
global_data_scale = .25  # (0 - 1) amount of data from the input data set to be trained on (scaling the train set down)
random_state = 101  # random seed

# used fro pickling trained models
## NOTE: not full automated yet
pickle_jar_loc = "C:/Users/twmar/OneDrive/Documents/Software Dev/MLPickleJar/"

# PCA vars
PCA_total_steps = 3  # amount of different PCA iterations
PCA_min = 2  # smallest amount of features to be outputted by PCA

# comment out any scaler that should not be used
# set raw_data Bool on if the raw data should also be used (instead of scaled)
raw_data = True
scalers = [
    StandardScaler(),
    MinMaxScaler(),
    MaxAbsScaler(),
    RobustScaler(),
    Normalizer(),
    QuantileTransformer(),
    PowerTransformer(),
    raw_data
]