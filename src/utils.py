# -*- encoding: utf-8 -*-
"""
@Time    :   2025-07-01 13:32:06
@desc    :
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""

import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from openai import OpenAI
from pandas import DataFrame
from tqdm import trange


def get_paths() -> tuple[Path, Path]:
    # 获取数据集和输出文件的路径
    root_path = Path(__file__).parents[1]
    datap = root_path / "data"
    datasetp = datap / "dataset"
    result_path = datap / "result"
    result_path.mkdir(parents=True, exist_ok=True)
    result_file = result_path / "test_pred.csv"
    logger.info(f"数据集路径: {datasetp}")
    logger.info(f"输出文件路径: {result_file}")
    return datasetp, result_file


def load_data(datasetp):
    # 加载训练集和测试集，并输出类别信息
    logger.info("开始加载数据...")
    train = pd.read_csv(datasetp / "train_all.csv")
    test = pd.read_csv(datasetp / "test_text.csv")
    category_list = list(train["类别"].unique())
    logger.info(f"类别数: {len(category_list)}")
    logger.info(f"类别列表: {category_list}")
    return train, test, category_list


class BaseClassifier(ABC):
    # 分类器基类，定义接口
    def __init__(self, train: DataFrame, test: DataFrame):
        self.train = train
        self.test = test

    @abstractmethod
    def train_model(self):
        # 训练模型
        pass

    @abstractmethod
    def predict(self) -> Any:
        # 预测测试集
        pass

    @abstractmethod
    def cross_val_score(self) -> float:
        # 交叉验证评分
        pass


class QwenEmbedder:
    """
    Qwen3-Embedding-0.6B 封装类，API实现，提供向量化和相似度计算方法
    """

    def __init__(
        self,
        api_token: str = "sk-vojzcipipuswxioqonijqnsdinbleotzmvyizdlhorrinyip",
        cache_path: str = "qwen_embed_cache.pkl",
    ):
        self._api_url = "https://api.siliconflow.cn/v1"
        self._api_token = api_token
        self._model_name = "Qwen/Qwen3-Embedding-0.6B"
        self._cache_path = cache_path
        self._cache = self._load_cache()
        self._client = OpenAI(base_url=self._api_url, api_key=self._api_token)

    def _load_cache(self):
        if os.path.exists(self._cache_path):
            try:
                with open(self._cache_path, "rb") as f:
                    cache = pickle.load(f)
                return cache
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        try:
            with open(self._cache_path, "wb") as f:
                pickle.dump(self._cache, f)
        except Exception:
            pass

    def _get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f"Instruct: {task_description}\nQuery:{query}"

    def embed(self, texts, is_query=False):
        """
        向量化文本（通过API）
        :param texts: 文本或文本列表
        :param is_query: 是否为query（会加prompt）
        :return: 向量或向量列表 (np.ndarray)
        """
        single_input = False
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        # 处理query prompt
        if is_query:
            task = "Given a web search query, retrieve relevant passages that answer the query"
            texts = [self._get_detailed_instruct(task, t) for t in texts]

        # 检查cache
        cached_vecs = []
        uncached_texts = []
        uncached_indices = []
        for idx, t in enumerate(texts):
            if self._cache.get(t) is not None:
                cached_vecs.append((idx, self._cache[t]))
            else:
                uncached_texts.append(t)
                uncached_indices.append(idx)

        # 需要请求API的文本
        new_vecs = []
        cache_updated = False
        max_batch = 256  # 最大batch
        if uncached_texts:
            for batch_start in trange(0, len(uncached_texts), max_batch):
                batch_texts = uncached_texts[batch_start : batch_start + max_batch]
                # openai sdk 支持批量
                response = self._client.embeddings.create(
                    input=batch_texts, model=self._model_name
                )
                embeddings = [item.embedding for item in response.data]
                # 写入cache
                for t, emb in zip(batch_texts, embeddings):
                    self._cache[t] = np.array(emb)
                new_vecs.extend(embeddings)
                cache_updated = True

        # 按原顺序组装结果
        result_vecs = [None] * len(texts)
        for idx, vec in cached_vecs:
            result_vecs[idx] = vec
        for i, idx in enumerate(uncached_indices):
            result_vecs[idx] = np.array(new_vecs[i])

        if cache_updated:
            self._save_cache()

        if single_input:
            return result_vecs[0]
        return np.stack(result_vecs)

    def similarity(self, query_embeddings, document_embeddings):
        """
        计算相似度
        :param query_embeddings: 查询向量 (np.ndarray)
        :param document_embeddings: 文档向量 (np.ndarray)
        :return: 相似度矩阵 (np.ndarray)
        """
        q = np.atleast_2d(query_embeddings)
        d = np.atleast_2d(document_embeddings)
        return np.matmul(q, d.T)


embeder = QwenEmbedder()

if __name__ == "__main__":
    # 测试QwenEmbedder
    texts = ["你好，世界！", "这是一个测试1。"]
    embeddings = embeder.embed(texts)
    print("向量化结果：", embeddings)

    query_embeddings = embeder.embed("你好，世界！", is_query=True)
    print("查询向量：", query_embeddings)

    similarity = embeder.similarity(query_embeddings, embeddings)
    print("相似度：", similarity)
    embeddings = embeder.embed(texts)
    print("向量化结果：", embeddings)

    query_embeddings = embeder.embed("你好，世界！", is_query=True)
    print("查询向量：", query_embeddings)

    similarity = embeder.similarity(query_embeddings, embeddings)
    print("相似度：", similarity)
