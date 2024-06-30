
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from loguru import logger

from src.classifier import VecLlmClassifier
from src.utils.data_processing import load_toutiao_data

TEST_DATA_PATH = "data/toutiao_cat_data/test_set_20240629.txt"
test_data = load_toutiao_data(TEST_DATA_PATH)

vlc = VecLlmClassifier()
test_list = []
pred_list = []
labels = set()
for i in tqdm(range(len(test_data)), desc="RUNNING TEST"):
    test_list.append(test_data[i][1][5])
    labels.add(test_data[i][1][5])
    pred_list.append(vlc.predict(test_data[i][0]))
labels = list(labels)

logger.info("\n{}".format(classification_report(test_list, pred_list, labels = labels)))
logger.info("\n{}".format(confusion_matrix(test_list, pred_list, labels=labels)))