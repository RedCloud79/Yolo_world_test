from transformers import CLIPProcessor, CLIPModel, Trainer, TrainingArguments
from PIL import Image
import torch

# 모델 & 프로세서 로딩
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 데이터 전처리 함수
def preprocess(example):
    image = Image.open(example["image_path"]).convert("RGB")
    inputs = processor(text=example["text"], images=image, return_tensors="pt", padding=True)
    return {
        "pixel_values": inputs["pixel_values"][0],
        "input_ids": inputs["input_ids"][0],
        "attention_mask": inputs["attention_mask"][0]
    }

# 데이터셋 변환
dataset = dataset.map(preprocess)

# Trainer용 데이터 클래스
class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "pixel_values": item["pixel_values"],
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"]
        }

    def __len__(self):
        return len(self.dataset)

train_dataset = CLIPDataset(dataset)

# 학습 인자 설정
training_args = TrainingArguments(
    output_dir="./clip-finetuned",
    per_device_train_batch_size=4,
    num_train_epochs=5,
    logging_steps=10,
    save_steps=50,
    evaluation_strategy="no",
    remove_unused_columns=False,
    report_to="none"
)

# Trainer 정의
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# 학습 시작
trainer.train()


#### ( 데이터 정리 예시 )
# from datasets import Dataset

# # 예시 데이터
# data = {
#     "image_path": ["img1.jpg", "img2.jpg", "img3.jpg"],
#     "text": ["a photo of smoke", "thick smoke in the sky", "dense gray smoke"]
# }

# dataset = Dataset.from_dict(data)

####