# -*- encoding: utf-8 -*-
"""
@Time    :   2025-07-01 13:31:35
@desc    :
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""

import jieba
from loguru import logger
from pandas.core.frame import DataFrame
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.svm import LinearSVC

from src.utils import BaseClassifier, embeder


class TfidfSVCClassifier(BaseClassifier):
    # 使用TF-IDF和LinearSVC的分类器
    def __init__(self, train: DataFrame, test: DataFrame):
        super().__init__(train, test)
        from sklearn.feature_extraction.text import TfidfVectorizer

        logger.info("初始化 TfidfSVCClassifier ...")
        # 使用jieba分词
        self.vectorizer = TfidfVectorizer(tokenizer=jieba.lcut)
        self.model = LinearSVC()
        self.train_tfidf = self.vectorizer.fit_transform(self.train["文本"])  #

    def train_model(self):
        # 训练TF-IDF向量和SVM模型
        logger.info("训练 TfidfSVCClassifier 模型 ...")
        self.model.fit(self.train_tfidf, self.train["类别"])

    def predict(self):
        # 对测试集进行预测
        logger.info("TfidfSVCClassifier 进行预测 ...")
        test_tfidf = self.vectorizer.transform(self.test["文本"])
        return self.model.predict(test_tfidf)

    def cross_val_score(self):
        # 计算交叉验证F1分数
        logger.info("TfidfSVCClassifier 交叉验证评分 ...")
        y = self.train["类别"]
        pred = cross_val_predict(LinearSVC(), X=self.train_tfidf, y=y)
        return f1_score(y, pred, average="macro")


class EmbeddingSVCClassifier(BaseClassifier):
    # 使用QwenEmbedder向量和LinearSVC的分类器
    def __init__(self, train: DataFrame, test: DataFrame):
        super().__init__(train, test)
        from sklearn.svm import LinearSVC

        logger.info("初始化 EmbeddingSVCClassifier ...")
        # 使用QwenEmbedder向量化文本
        self.train_vec = embeder.embed(self.train["文本"].tolist())
        self.model = LinearSVC()

    def train_model(self):
        logger.info("训练 EmbeddingSVCClassifier 模型 ...")
        self.model.fit(self.train_vec, self.train["类别"])

    def predict(self):
        logger.info("EmbeddingSVCClassifier 进行预测 ...")
        test_vec = embeder.embed(self.test["文本"].tolist())
        return self.model.predict(test_vec)

    def cross_val_score(self):
        from sklearn.metrics import f1_score
        from sklearn.model_selection import cross_val_predict

        logger.info("EmbeddingSVCClassifier 交叉验证评分 ...")
        y = self.train["类别"]
        pred = cross_val_predict(self.model, X=self.train_vec, y=y)
        return f1_score(y, pred, average="macro")


# 使用向量模型表示语义进行分类
