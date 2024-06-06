
from tasks.superglue.dataset import task_to_keys as superglue_tasks


SUPERGLUE_DATASETS = list(superglue_tasks.keys())
NER_DATASETS = ["conll2003", "conll2004"]


TASKS = ["superglue", "ner"]

DATASETS = SUPERGLUE_DATASETS + NER_DATASETS
