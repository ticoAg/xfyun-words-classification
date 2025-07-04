import datetime
import sys
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.nn import functional as F
from tqdm import trange
from transformers import AutoTokenizer, BertForSequenceClassification
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

import wandb

sys.path.append(Path(__file__).parents[2].as_posix())  # 添加src目录到路径中
from src.utils import BaseClassifier  # 假设base.py在src目录下


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if self.class_weights is not None:
            weights = torch.tensor(
                self.class_weights, device=logits.device, dtype=logits.dtype
            )
            loss = F.cross_entropy(logits, labels, weight=weights)
        else:
            loss = F.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss


class BertBaseClassifier(BaseClassifier):
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        super().__init__(train, test)
        self.model_name = "hfl/chinese-macbert-large"
        self.max_length = 256
        self.batch_size = 4
        self.epochs = 5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.le = LabelEncoder()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = None
        self.trainer = None
        self.train_dataset = None
        self.test_dataset = None
        # 初始化时直接从train获取类别并fit
        if train is not None and "类别" in train.columns:
            self.label_classes = train["类别"].unique()
            self.le.fit(self.label_classes)

    def preprocess(self, examples):
        return self.tokenizer(
            examples["文本"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro")
        return {"accuracy": acc, "macro_f1": macro_f1}

    def split_data(self):
        train_df, val_df = train_test_split(
            self.train, test_size=0.15, random_state=42, stratify=self.train["label"]
        )
        train_df = pd.DataFrame(train_df)
        val_df = pd.DataFrame(val_df)
        logger.info("训练集类别分布:\n%s" % train_df["类别"].value_counts().to_dict())
        logger.info("验证集类别分布:\n%s" % val_df["类别"].value_counts().to_dict())
        return train_df, val_df

    def oversample_minority(self, train_df):
        """
        对少数类进行过采样，确保每个类别的样本数量大致相同。
        使用动态阈值来决定过采样的目标数量。
        """

        label_counts = train_df["label"].value_counts()
        max_count = label_counts.max()

        # 动态调整阈值：使用中位数或分位数
        median = label_counts.median()
        threshold = int(
            max(median * 0.7, max_count * 0.1)
        )  # 取中位数70%和最大类10%的较高者

        dfs = []
        for label, count in label_counts.items():
            df_label = train_df[train_df["label"] == label]

            # 对中等少数类部分增强
            if count < max_count * 0.3:  # 扩展处理范围
                target = min(int(threshold * 1.5), max_count)  # 控制上限
                if count < target:
                    # 这里原本是 df_label = df_label.sample(threshold, ...)
                    # 但应该采样到 target 数量
                    df_label = df_label.sample(target, replace=True, random_state=42)
                    # 添加数据增强方法
                    # df_label = augment_samples(df_label, target_count=target)
            elif count < threshold:
                df_label = df_label.sample(threshold, replace=True, random_state=42)
            dfs.append(df_label)

        resampled_df = pd.concat(dfs).sample(frac=1, random_state=42)
        logger.info(
            "过采样后训练集类别分布:\n%s"
            % resampled_df["类别"].value_counts().to_dict()
        )
        return resampled_df

    def build_datasets(self, train_df, val_df):
        train_dataset = Dataset.from_pandas(pd.DataFrame(train_df[["文本", "label"]]))
        train_dataset = train_dataset.map(self.preprocess, batched=True)
        val_dataset = Dataset.from_pandas(pd.DataFrame(val_df[["文本", "label"]]))
        val_dataset = val_dataset.map(self.preprocess, batched=True)
        return train_dataset, val_dataset

    def compute_class_weights(self, train_df):
        class_counts = train_df["label"].value_counts().sort_index().values
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        return class_weights

    def init_wandb(self):
        run_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project="macbert_classify", name=run_name, reinit=True)

    def get_resume_checkpoint(self, output_dir="./ckpts"):
        checkpoints = list(Path(output_dir).glob("checkpoint-*"))
        resume_from_checkpoint = None
        if checkpoints:
            checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            resume_from_checkpoint = str(checkpoints[0])
            logger.info(f"发现检查点: {resume_from_checkpoint}，将继续训练")
        return resume_from_checkpoint

    def train_model(self):
        # wandb初始化
        self.init_wandb()
        # 检查是否存在之前的检查点
        resume_from_checkpoint = self.get_resume_checkpoint()

        # 标签编码 - 确保使用所有类别的完整集合
        self.label_classes = self.train["类别"].unique()  # 存储所有类别
        self.le.fit(self.label_classes)  # 使用完整类别集合拟合编码器
        self.train["label"] = self.le.transform(self.train["类别"])

        # 检查标签类别数量
        if self.le.classes_ is not None:
            num_labels = len(self.le.classes_)
        else:
            raise ValueError(
                "LabelEncoder.classes_ is None. Please fit the encoder before using it."
            )
        logger.info(f"标签类别数量: {num_labels}")

        # 数据处理流程
        train_df, val_df = self.split_data()
        train_df = self.oversample_minority(train_df)
        self.train_dataset, self.val_dataset = self.build_datasets(train_df, val_df)
        class_weights = self.compute_class_weights(train_df)

        # 模型
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=num_labels
        )
        self.model.to(self.device)  # type: ignore
        # 训练参数
        training_args = TrainingArguments(
            output_dir="./ckpts",
            per_device_train_batch_size=self.batch_size,  # 增加批次大小
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=5,  # 增加训练轮数
            save_strategy="steps",
            save_steps=500,  # 更频繁地保存检查点
            logging_steps=25,  # 更细粒度地监控训练过程
            eval_strategy="steps",
            eval_steps=500,  # 增加验证频率
            logging_dir="./logs",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            report_to=["wandb"],
            save_total_limit=3,
            learning_rate=2e-5,  # 降低学习率
            max_grad_norm=1.0,  # 添加梯度裁剪
            warmup_ratio=0.15,  # 添加学习率预热
            weight_decay=0.01,  # 添加权重衰减
            resume_from_checkpoint=resume_from_checkpoint,  # 从检查点恢复训练
        )
        # Trainer
        self.trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
            class_weights=class_weights,
        )
        # 训练
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        wandb.finish()

    def load_model_from_checkpoint(self, checkpoint_path=None):
        """
        加载指定checkpoint权重，或自动查找./ckpts下最新checkpoint，若都没有则加载原始模型。
        """
        ckpt_to_load = None
        if checkpoint_path is not None:
            ckpt_to_load = checkpoint_path
        else:
            checkpoints = list(Path("./ckpts").glob("checkpoint-*/"))
            if checkpoints:
                checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                ckpt_to_load = str(checkpoints[0])
        if ckpt_to_load is not None:
            logger.info(f"加载权重: {ckpt_to_load}")
            self.model = BertForSequenceClassification.from_pretrained(
                ckpt_to_load, num_labels=len(self.le.classes_)
            ).to(self.device)
        else:
            logger.warning("未找到训练权重，加载原始预训练模型")
            self.model = BertForSequenceClassification.from_pretrained(
                self.model_name, num_labels=len(self.le.classes_)
            ).to(self.device)

    def predict(self, checkpoint_path=None):
        """
        优先加载指定checkpoint权重，否则自动查找./ckpts下最新checkpoint。
        若都没有，则回退加载原始模型。
        """

        if self.model is None:
            self.load_model_from_checkpoint(checkpoint_path)

        # 测试集格式转换
        self.test_dataset = Dataset.from_pandas(self.test.loc[:, ["文本"]])
        self.test_dataset = self.test_dataset.map(self.preprocess, batched=True)
        preds = []
        self.model.eval()  # type: ignore
        bs = self.batch_size * 4
        with torch.no_grad():
            for i in trange(0, len(self.test_dataset), bs):
                batch = self.test_dataset[i : i + bs]
                inputs = {
                    k: torch.tensor(batch[k]).to(self.device)
                    for k in ["input_ids", "attention_mask"]
                }
                outputs = self.model(**inputs)
                pred = outputs.logits.argmax(dim=-1).cpu().numpy()
                preds.extend(pred)
        # 反编码
        self.test["类别"] = self.le.inverse_transform(preds)
        return self.test[["id", "类别"]]

    def cross_val_score(self) -> float:
        # 由于BERT等大模型训练开销较大，且transformers的Trainer不直接支持sklearn风格的交叉验证，
        # 实现交叉验证需要多次重新加载和训练模型，效率较低，代码复杂度也会提升。
        # 因此此处未实现交叉验证，仅返回-1作为占位。
        return -1.0


def augment_samples(df, target_count):
    """
    对DataFrame中的文本进行EDA增强，扩充到target_count。
    每条原始文本直接用eda.eda(text, num_aug=4)生成4条增强样本。
    """
    rows = []
    texts = df["文本"].tolist()
    label = df["label"].iloc[0]
    n = len(df)
    # 先保留原始
    rows.extend(df.to_dict(orient="records"))
    # 逐条增强
    i = 0
    while len(rows) < target_count:
        text = texts[i % n]
        aug_set = eda.eda(text, num_aug=4)
        for aug_text in aug_set:
            rows.append({"文本": aug_text, "label": label})
            if len(rows) >= target_count:
                break
        i += 1
    return pd.DataFrame(rows[:target_count])


if __name__ == "__main__":
    from src.utils import get_paths, load_data

    datasetp, output_file = get_paths()
    train, test, category_list = load_data(datasetp)

    clf = BertBaseClassifier(train, test)
    # clf.train_model()
    result = clf.predict()
    result.to_csv(output_file, index=False)
    logger.info(f"预测完成，结果已保存到{output_file}")
