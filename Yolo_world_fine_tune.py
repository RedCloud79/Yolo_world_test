_base_ = ['../_base_/default_runtime.py']

model = dict(
    type='YOLOWorld',
    clip_adapter=dict(
        type='CLIPAdapterLoRA',
        vision_adapter=dict(lora_rank=4),
        text_adapter=dict(lora_rank=4),
    ),
)

dataset_type = 'YOLOv5Dataset'
data_root = 'your_dataset/'

train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/labels/',
        img_prefix='train/images/',
        classes=['fire', 'smoke'],
    )
)

val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val/labels/',
        img_prefix='val/images/',
        classes=['fire', 'smoke'],
    )
)
