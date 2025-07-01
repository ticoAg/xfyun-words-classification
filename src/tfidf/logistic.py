# -*- encoding: utf-8 -*-
"""
@Time    :   2025-07-01 13:32:35
@desc    :
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""

import jieba
from loguru import logger
from pandas.core.frame import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict

from src.utils import BaseClassifier


class TfidfLogisticClassifier(BaseClassifier):
    # 使用TF-IDF和LogisticRegression的分类器
    def __init__(self, train: DataFrame, test: DataFrame):
        super().__init__(train, test)
        from sklearn.feature_extraction.text import TfidfVectorizer

        logger.info("初始化 TfidfLogisticClassifier ...")
        self.vectorizer = TfidfVectorizer(tokenizer=jieba.lcut)
        self.model = LogisticRegression(max_iter=1000)
        self.train_tfidf = self.vectorizer.fit_transform(self.train["文本"])

    def train_model(self):
        # 训练TF-IDF向量和Logistic回归模型
        logger.info("训练 TfidfLogisticClassifier 模型 ...")
        self.model.fit(self.train_tfidf, self.train["类别"])

    def predict(self):
        # 对测试集进行预测
        logger.info("TfidfLogisticClassifier 进行预测 ...")
        test_tfidf = self.vectorizer.transform(self.test["文本"])
        return self.model.predict(test_tfidf)

    def cross_val_score(self):
        # 计算交叉验证F1分数
        logger.info("TfidfLogisticClassifier 交叉验证评分 ...")
        y = self.train["类别"]
        pred = cross_val_predict(
            LogisticRegression(max_iter=1000), X=self.train_tfidf, y=y
        )
        return f1_score(y, pred, average="macro")
