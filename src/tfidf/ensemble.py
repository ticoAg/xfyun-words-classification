# -*- encoding: utf-8 -*-
"""
@Time    :   2025-07-01 13:35:05
@desc    :
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""

import jieba
from loguru import logger
from pandas.core.frame import DataFrame
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.svm import LinearSVC

from src.utils import BaseClassifier


class TfidfVotingClassifier(BaseClassifier):
    # 集成方案：TF-IDF特征+多模型投票
    def __init__(self, train: DataFrame, test: DataFrame):
        super().__init__(train, test)
        from sklearn.feature_extraction.text import TfidfVectorizer

        logger.info("初始化 TfidfVotingClassifier ...")
        self.vectorizer = TfidfVectorizer(tokenizer=jieba.lcut)
        self.train_tfidf = self.vectorizer.fit_transform(self.train["文本"])
        self.model = VotingClassifier(
            estimators=[
                ("svc", LinearSVC()),
                ("lr", LogisticRegression(max_iter=1000)),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=100, random_state=42, n_jobs=-1, verbose=1
                    ),
                ),
            ],
            voting="hard",
        )

    def train_model(self):
        # 训练TF-IDF向量和投票分类器模型
        logger.info("训练 TfidfVotingClassifier 模型 ...")
        self.model.fit(self.train_tfidf, self.train["类别"])

    def predict(self):
        # 对测试集进行预测
        logger.info("TfidfVotingClassifier 进行预测 ...")
        test_tfidf = self.vectorizer.transform(self.test["文本"])
        return self.model.predict(test_tfidf)

    def cross_val_score(self):
        # 计算交叉验证F1分数
        logger.info("TfidfVotingClassifier 交叉验证评分 ...")
        y = self.train["类别"]
        pred = cross_val_predict(self.model, X=self.train_tfidf, y=y)
        return f1_score(y, pred, average="macro")
