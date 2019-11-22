import utilities
import logging
import numpy as np

# Settings. Read from JSON later on
settings = {
    # Data Processing
    "orig_train_path": u"data/processed/train.csv",
    "orig_test_path": u"data/processed/test.csv",
    "orig_output_path": u"data/processed/data.csv",

    # Miscellaneous
    "seed": 42,
    "log_level": logging.INFO,
    "use_cache": True
}

# Defining Logging
logging.basicConfig(level=settings["log_level"])
logger = logging.getLogger(__name__)

# Settings a global seed
np.random.seed(settings["seed"])

# Prepare the original data, as it is divided in train and test
utilities.concatenate_data(settings)
train, val, test, stest = utilities.create_data_sets(settings)

# Pre Processing

# Modelling

# Training

# Validating


