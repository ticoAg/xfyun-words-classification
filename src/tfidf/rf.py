# -*- encoding: utf-8 -*-
"""
@Time    :   2025-07-01 13:34:25
@desc    :
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""

import jieba
from loguru import logger
from pandas.core.frame import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict

from src.utils import BaseClassifier


class TfidfRFClassifier(BaseClassifier):
    # 使用TF-IDF和RandomForest的分类器
    def __init__(self, train: DataFrame, test: DataFrame):
        super().__init__(train, test)
        from sklearn.feature_extraction.text import TfidfVectorizer

        logger.info("初始化 TfidfRFClassifier ...")
        self.vectorizer = TfidfVectorizer(tokenizer=jieba.lcut)
        self.model = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1, verbose=1
        )
        self.train_tfidf = self.vectorizer.fit_transform(self.train["文本"])

    def train_model(self):
        # 训练TF-IDF向量和随机森林模型
        logger.info("训练 TfidfRFClassifier 模型 ...")
        self.model.fit(self.train_tfidf, self.train["类别"])

    def predict(self):
        # 对测试集进行预测
        logger.info("TfidfRFClassifier 进行预测 ...")
        test_tfidf = self.vectorizer.transform(self.test["文本"])
        return self.model.predict(test_tfidf)

    def cross_val_score(self):
        # 计算交叉验证F1分数
        logger.info("TfidfRFClassifier 交叉验证评分 ...")
        y = self.train["类别"]
        pred = cross_val_predict(
            RandomForestClassifier(n_estimators=100, random_state=42),
            X=self.train_tfidf,
            y=y,
            n_jobs=-1,
            verbose=1,
        )
        return f1_score(y, pred, average="macro")
