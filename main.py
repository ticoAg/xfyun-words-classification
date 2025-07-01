# -*- encoding: utf-8 -*-
"""
@Time    :   2025-07-01 13:36:51
@desc    :
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""

from datetime import datetime
from pathlib import Path
from typing import Type

import pandas as pd
from loguru import logger

from src.tfidf import (
    EmbeddingSVCClassifier,
    TfidfLogisticClassifier,
    TfidfRFClassifier,
    TfidfSVCClassifier,
    TfidfVotingClassifier,
)
from src.utils import BaseClassifier, get_paths, load_data


def run_classifier(classifier_cls: Type[BaseClassifier], train, test, output_file):
    # 运行指定的分类器，输出预测结果
    logger.info(f"运行分类器: {classifier_cls.__name__}")
    clf = classifier_cls(train, test)
    clf.train_model()
    score = clf.cross_val_score()
    logger.info(f"{classifier_cls.__name__} F1 Score: {score:.4f}")
    pred = clf.predict()
    output = pd.DataFrame({"id": test.index, "类别": pred})
    # 将score格式化为xx_xx（如0.5846 -> 58_46），用于文件名
    score_str = f"{score * 100:.2f}".replace(".", "_")
    # 获取时间戳
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 构造新文件名：test_pred_svc.58_46.20250701_134809.csv
    output_file_with_score = output_file.with_name(
        f"{output_file.stem}.{score_str}.{time_str}{output_file.suffix}"
    )
    output.to_csv(output_file_with_score, index=False, encoding="utf-8")
    logger.info(f"预测结果已保存到: {output_file_with_score}")
    return score, pred


def main():
    # 主流程：加载数据、选择分类器并运行
    logger.info("程序开始运行")
    datasetp, output_file = get_paths()
    train, test, category_list = load_data(datasetp)

    # 依次输出所有方案的得分和结果
    classifiers = [
        (TfidfSVCClassifier, "svc", "=== SVC 方案 ==="),
        (EmbeddingSVCClassifier, "embedding_svc", "=== Embedding SVC 方案 ==="),
        (TfidfLogisticClassifier, "logistic", "=== Logistic 方案 ==="),
        (TfidfRFClassifier, "rf", "=== RandomForest 方案 ==="),
        (TfidfVotingClassifier, "voting", "=== 集成方案（Voting） ==="),
    ]
    for clf_cls, fname, title in classifiers:
        logger.info(title)
        output_path = Path(output_file)
        output_file_new = output_path.with_name(f"{output_path.stem}_{fname}.csv")
        run_classifier(clf_cls, train, test, output_file_new)

    logger.info("程序运行结束")


if __name__ == "__main__":
    main()
