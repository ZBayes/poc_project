
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from loguru import logger

from src.classifier import VecLlmClassifier
from src.utils.data_processing import load_toutiao_data

VERSION = "20240702_FEW"
TEST_DATA_PATH = "data/toutiao_cat_data/test_set_{}.txt".format(VERSION)
OUTPUT_DATA_PATH = "data/toutiao_cat_data/test_set_{}_result.txt".format(VERSION)
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

with open(OUTPUT_DATA_PATH, "w", encoding="utf8") as fout:
    for idx in range(len(test_data)):
        fout.write("{}\t{}\t{}\t{}\n".format(test_data[idx][0], 
                                            test_list[idx],
                                            pred_list[idx],
                                            test_list[idx]==pred_list[idx]))