import wandb
from config.keys import *


def init_wandb(project_name, entity=None):
    wandb.login(key=WANDB_API_KEY)
    wandb.init(project=project_name, entity=entity)


PROJECT_NAME = "amazon-lt-collection"
ENTITY = None  # Optional: specify the entity if needed

# init_wandb(PROJECT_NAME, ENTITY)