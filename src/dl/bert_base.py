import datetime
import sys
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, BertForSequenceClassification
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

import wandb

sys.path.append(Path(__file__).parents[2].as_posix())  # 添加src目录到路径中
from src.utils import BaseClassifier  # 假设base.py在src目录下


class BertBaseClassifier(BaseClassifier):
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        super().__init__(train, test)
        self.model_name = "hfl/chinese-macbert-large"
        self.max_length = 256
        self.batch_size = 8
        self.epochs = 3
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.le = LabelEncoder()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = None
        self.trainer = None
        self.train_dataset = None
        self.test_dataset = None

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

    def train_model(self):
        # wandb初始化
        run_name = f"macbert_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project="macbert_classify", name=run_name, reinit=True)
        # 标签编码
        self.train["label"] = self.le.fit_transform(self.train["类别"])
        # 划分训练集和验证集
        train_df, val_df = train_test_split(
            self.train,
            test_size=0.15,
            random_state=42,
            stratify=self.train["label"],
        )
        self.train_dataset = Dataset.from_pandas(train_df.loc[:, ["文本", "label"]])
        self.train_dataset = self.train_dataset.map(self.preprocess, batched=True)
        self.val_dataset = Dataset.from_pandas(val_df.loc[:, ["文本", "label"]])
        self.val_dataset = self.val_dataset.map(self.preprocess, batched=True)
        # 模型
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(self.le.classes_)
        ).to(self.device)
        # 训练参数
        training_args = TrainingArguments(
            output_dir="./macbert_baseline_ckpt",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            save_strategy="steps",  # 修正为steps
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=500,
            logging_dir="./logs",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            report_to=["wandb"],
        )
        # Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
        )
        # 训练
        self.trainer.train()
        wandb.finish()

    def predict(self):
        # 如果模型未加载，则加载模型
        if self.model is None:
            self.model = BertForSequenceClassification.from_pretrained(
                self.model_name, num_labels=len(self.le.classes_)
            ).to(self.device)
            # 加载训练好的权重（如果有保存的话，可以在此处加载权重文件）

        # 测试集格式转换
        self.test_dataset = Dataset.from_pandas(self.test.loc[:, ["文本"]])
        self.test_dataset = self.test_dataset.map(self.preprocess, batched=True)
        preds = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(self.test_dataset), self.batch_size):
                batch = self.test_dataset[i : i + self.batch_size]
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


if __name__ == "__main__":
    from src.utils import get_paths, load_data

    datasetp, output_file = get_paths()
    train, test, category_list = load_data(datasetp)

    clf = BertBaseClassifier(train, test)
    clf.train_model()
    result = clf.predict()
    result.to_csv("macbert_wwm_baseline_predict.csv", index=False)
    print("预测完成，结果已保存到macbert_wwm_baseline_predict.csv")
