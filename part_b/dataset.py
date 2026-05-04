# import os
# import pandas as pd
# from datasets import Dataset, Audio, Features, Value
# from sklearn.model_selection import train_test_split
# from dataclasses import dataclass
# from typing import Any, Dict, List, Union
# import torch

# class AzerbaijanSpeechDataset:
#     def __init__(self, config, processor):
#         self.config = config
#         self.processor = processor

#     def load_and_prepare(self):
#         # 1. TSV faylını oxuyuruq
#         df = pd.read_csv(self.config.TSV_PATH, sep="\t")[["path", "sentence"]].dropna()
#         df = df.sample(min(len(df), self.config.NUM_SAMPLES), random_state=42).reset_index(drop=True)
        
#         # 2. Audio yollarını hazırlayırıq
#         df["audio"] = df["path"].apply(lambda x: os.path.abspath(os.path.join(self.config.AUDIO_DIR, x)).replace("\\", "/"))

#         # 3. Train/Test split
#         train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
#         # 4. Əsas düzəliş: Dataframe-i lüğətə çeviririk (Index problemlərindən qaçmaq üçün)
#         train_dict = train_df.to_dict('list')
#         test_dict = test_df.to_dict('list')

#         # 5. Features strukturu
#         features = Features({
#             "audio": Audio(sampling_rate=16000),
#             "sentence": Value("string"),
#             "path": Value("string")
#         })

#         # 6. Lüğətdən dataset yaradırıq (Bu daha stabil üsuldur)
#         train_ds = Dataset.from_dict(train_dict, features=features)
#         test_ds = Dataset.from_dict(test_dict, features=features)

#         # 7. Map əməliyyatı
#         train_ds = train_ds.map(self.prepare_batch, remove_columns=train_ds.column_names)
#         test_ds = test_ds.map(self.prepare_batch, remove_columns=test_ds.column_names)

#         return train_ds, test_ds

#     def prepare_batch(self, batch):
#         audio = batch["audio"]
#         batch["input_features"] = self.processor.feature_extractor(
#             audio["array"], 
#             sampling_rate=16000
#         ).input_features[0]
        
#         batch["labels"] = self.processor.tokenizer(batch["sentence"].lower()).input_ids
#         return batch

# @dataclass
# class DataCollatorSpeechSeq2SeqWithPadding:
#     processor: Any
#     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
#         input_features = [{"input_features": feature["input_features"]} for feature in features]
#         batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
#         label_features = [{"input_ids": feature["labels"]} for feature in features]
#         labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
#         labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
#         batch["labels"] = labels
#         return batch


import os
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
import librosa


class AzerbaijanSpeechDataset:
    def __init__(self, config, processor):
        self.config = config
        self.processor = processor

    def load_and_prepare(self):

        # 1. Load TSV
        df = pd.read_csv(self.config.TSV_PATH, sep="\t")[["path", "sentence"]].dropna()

        df = df.sample(
            min(len(df), self.config.NUM_SAMPLES),
            random_state=42
        ).reset_index(drop=True)

        # 2. Audio path (NO datasets.Audio anymore)
        df["audio"] = df["path"].apply(
            lambda x: os.path.join(self.config.AUDIO_DIR, x)
        )

        # 3. Train/Test split
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42
        )

        # 4. Convert to dict
        train_dict = train_df.to_dict("list")
        test_dict = test_df.to_dict("list")

        # 5. Create datasets (NO Audio feature)
        train_ds = Dataset.from_dict(train_dict)
        test_ds = Dataset.from_dict(test_dict)

        # 6. Map preprocessing
        train_ds = train_ds.map(
            self.prepare_batch,
            remove_columns=train_ds.column_names
        )

        test_ds = test_ds.map(
            self.prepare_batch,
            remove_columns=test_ds.column_names
        )

        return train_ds, test_ds

    # =========================================================
    # SAFE AUDIO LOADING (FIX FOR TORCHCODEC ERROR)
    # =========================================================
    def load_audio(self, path):
        speech, sr = librosa.load(path, sr=16000)

        # normalize
        speech = speech / (abs(speech).max() + 1e-9)

        return speech

    # =========================================================
    # PREPROCESSING
    # =========================================================
    def prepare_batch(self, batch):

        audio_path = batch["audio"]

        speech = self.load_audio(audio_path)

        batch["input_features"] = self.processor.feature_extractor(
            speech,
            sampling_rate=16000
        ).input_features[0]

        batch["labels"] = self.processor.tokenizer(
            batch["sentence"].lower()
        ).input_ids

        return batch


# =========================================================
# DATA COLLATOR (UNCHANGED - CORRECT)
# =========================================================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        input_features = [
            {"input_features": feature["input_features"]}
            for feature in features
        ]

        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )

        label_features = [
            {"input_ids": feature["labels"]}
            for feature in features
        ]

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100
        )

        batch["labels"] = labels

        return batch