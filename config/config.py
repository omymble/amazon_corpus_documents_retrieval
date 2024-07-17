import os
import torch

RANDOM_STATE = 42

# data constants
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(ROOT_DIR, 'data/')

# initial files
LABELLED_DATA = os.path.join(DATA_PATH, 'labelled_data/')
BOOKS_DATA = os.path.join(DATA_PATH, 'amazon_lt_collection_small/')

# preprocessed files
OBTAINED_DATA = os.path.join(DATA_PATH, 'obtained_data/')
SPLIT_REQUESTS = os.path.join(OBTAINED_DATA, 'split_requests/')
ABSA_PREPROCESSED = os.path.join(OBTAINED_DATA, 'absa_aspects_and_categories.pkl')


# saved models
MODELS = os.path.join(DATA_PATH, 'models')

# commons
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MAX_LEN = 128
BATCH_SIZE = 16
