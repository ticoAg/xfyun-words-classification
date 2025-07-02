# -*- encoding: utf-8 -*-
"""
@Time    :   2025-07-02 12:38:34
@desc    :
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger
from openai import AsyncOpenAI
from pandas.core.frame import DataFrame

sys.path.append(Path(__file__).parents[2].as_posix())  # 添加src目录到路径中

from dotenv import load_dotenv

load_dotenv()
from src.utils import BaseClassifier

aclient = AsyncOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY")
)


class OpenAIPromptClassifier(BaseClassifier):
    """
    基于OpenAI LLM的Prompt分类器
    """

    def __init__(
        self,
        train: DataFrame,
        test: DataFrame,
        api_key: Optional[str] = None,
        model: str = "doubao-seed-1-6-flash-250615",
    ):
        super().__init__(train, test)
        self.model = model
        if not aclient.api_key:
            raise ValueError(
                "OpenAI API key must be provided via argument or OPENAI_API_KEY env variable."
            )
        self.labels: list[str] = sorted(set(self.train["类别"]))
        self.label_map = {str(i): label for i, label in enumerate(self.labels)}
        self.label_str = "\n".join(
            [f"{i}. {label}" for i, label in self.label_map.items()]
        )
        logger.info(f"OpenAIPromptClassifier 初始化，类别: {self.labels}")

    def train_model(self):
        logger.info("OpenAIPromptClassifier 无需训练，直接调用LLM进行推理。")
        # 新增：交叉验证流程
        score = self.cross_val_score(sample_per_label=50)
        logger.info(f"交叉验证得分: {score}")

    def _build_prompt(self, text: str) -> str:
        prompt = (
            f"请根据以下文本内容，将其分类到如下类别之一:\n"
            f"{self.label_str}\n"
            f"要求: 只返回类别序号，不要返回其他内容\n"
            f"文本：{text}"
        )
        return prompt

    async def create_openai_completion(self, prompt: str) -> str:
        response = await aclient.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            # temperature=1,
            # extra_body={"enable_thinking": False},
            extra_body={
                "thinking": {
                    "type": "disabled"  # 不使用深度思考能力
                }
            },
        )
        if response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        else:
            return ""

    async def _classify(self, text: str) -> str:
        prompt = self._build_prompt(text)
        try:
            result = await self.create_openai_completion(prompt)
            if self.label_map.get(result):
                return self.label_map[result]
            else:
                logger.warning(f"LLM返回未知类别: {result}，将返回原始内容。")
            return result
        except Exception as e:
            logger.error(f"OpenAI API 调用失败: {e}")
            return "未知"

    async def _predict_all(self, texts, max_concurrency=1):
        from tqdm.asyncio import tqdm as async_tqdm

        semaphore = asyncio.Semaphore(max_concurrency)
        results = []

        async def sem_classify(text):
            async with semaphore:
                return await self._classify(text)

        tasks = [sem_classify(text) for text in texts]
        results = []
        for coro in async_tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="LLM推理进度"
        ):
            result = await coro
            results.append(result)
        return results

    def predict(self, max_concurrency=10):
        logger.info("OpenAIPromptClassifier 进行预测 ...")
        texts = self.test["文本"].tolist()
        results = asyncio.run(self._predict_all(texts, max_concurrency=max_concurrency))
        return results

    def cross_val_score(
        self, max_concurrency=10, sample_per_label=None, random_state=42
    ):
        """
        用训练集直接验证，支持对每个label平均采样，返回准确率和macro F1。
        sample_per_label: int or None，每个类别采样数量，None则用全部数据。
        """
        import numpy as np
        from sklearn.metrics import f1_score

        df = self.train.copy()
        if sample_per_label is not None:
            # 按类别分组采样
            df = (
                df.groupby("类别", group_keys=False)
                .apply(
                    lambda x: x.sample(
                        n=min(sample_per_label, len(x)), random_state=random_state
                    )
                )
                .reset_index(drop=True)
            )
        X = df["文本"].tolist()
        y = df["类别"].tolist()
        preds = asyncio.run(self._predict_all(X, max_concurrency=max_concurrency))
        acc = np.mean([p == t for p, t in zip(preds, y)])
        f1 = f1_score(y, preds, average="macro", labels=self.labels)
        logger.info(f"采样验证准确率: {acc:.4f}，macro F1: {f1:.4f}，样本数: {len(y)}")
        return f1


if __name__ == "__main__":
    from src.utils import get_paths, load_data

    datasetp, output_file = get_paths()
    train, test, category_list = load_data(datasetp)
    classifier = OpenAIPromptClassifier(train, test)
    classifier.train_model()
    results = classifier.predict()
    print(results)
