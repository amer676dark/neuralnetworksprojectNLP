"""
Wav2Vec 2.0 Arabic ASR — fine-tuning and inference wrapper.

Uses facebook/wav2vec2-large-xlsr-53-arabic from HuggingFace,
which is pre-trained on multilingual data and fine-tuned for Arabic.

For fine-tuning on your own data, use Wav2Vec2ForCTC with CTC loss
(same approach as the CNN+LSTM but with a powerful pretrained encoder).
"""

import torch
import numpy as np
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple
import numpy as np


# Best Arabic Wav2Vec2 models on HuggingFace
ARABIC_MODELS = {
    "xlsr-arabic": "facebook/wav2vec2-large-xlsr-53-arabic",
    "arabic-large": "othrif/wav2vec2-large-arabic",
    "arabic-base": "elgeish/wav2vec2-large-xlsr-53-arabic",
}


class Wav2Vec2ASR:
    """Wrapper for Wav2Vec2 Arabic ASR — supports inference and fine-tuning."""

    def __init__(
        self,
        # facebook/wav2vec2-large-xlsr-53-arabic was deprecated by FAIR;
        # using the community-maintained equivalent with same XLSR-53 backbone.
        model_name: str = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
        device: Optional[str] = None,
    ):
        self.model_name = model_name

        if device == "auto" or device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Loading Wav2Vec2 ({model_name}) on {self.device}...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("Wav2Vec2 model loaded.")

    def transcribe_array(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
    ) -> str:
        """Transcribe a numpy audio array (float32, mono, 16kHz)."""
        inputs = self.processor(
            waveform.astype(np.float32),
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            logits = self.model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription.strip()

    def transcribe_batch(
        self,
        waveforms: List[np.ndarray],
        sample_rate: int = 16000,
    ) -> List[str]:
        """Batch transcription for efficiency."""
        waveforms = [w.astype(np.float32) for w in waveforms]
        inputs = self.processor(
            waveforms,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        with torch.no_grad():
            logits = self.model(input_values, attention_mask=attention_mask).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(predicted_ids)

    def evaluate_dataset(
        self,
        dataset,
        batch_size: int = 8,
    ) -> Dict:
        """Evaluate on a HuggingFace dataset split, return WER/CER."""
        from utils.metrics import compute_batch_metrics, format_metrics_report

        all_refs, all_hyps = [], []

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i: i + batch_size]
            waveforms = [s["array"] for s in batch["audio"]]
            references = batch["sentence"]
            hypotheses = self.transcribe_batch(waveforms)
            all_refs.extend(references)
            all_hyps.extend(hypotheses)

        metrics = compute_batch_metrics(all_refs, all_hyps)
        print(format_metrics_report(metrics))
        return metrics, all_hyps


@dataclass
class DataCollatorCTCWithPadding:
    """Data collator for CTC fine-tuning — pads input_values and labels separately."""

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


def prepare_dataset_for_wav2vec(
    hf_dataset,
    processor: Wav2Vec2Processor,
    max_duration: float = 10.0,
) -> Dataset:
    """Preprocess HuggingFace dataset for Wav2Vec2 fine-tuning."""

    def preprocess(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        with processor.as_target_processor():
            batch["labels"] = processor(batch["sentence"]).input_ids
        return batch

    dataset = hf_dataset.map(
        preprocess,
        remove_columns=hf_dataset.column_names,
        num_proc=4,
        desc="Preprocessing for Wav2Vec2",
    )

    # Filter too-long samples
    max_samples = int(max_duration * 16000)
    dataset = dataset.filter(
        lambda x: x["input_length"] <= max_samples,
        desc="Filtering long audio",
    )
    return dataset


def fine_tune_wav2vec(
    model_name: str,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    output_dir: str = "./outputs/checkpoints/wav2vec",
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    warmup_ratio: float = 0.1,
) -> None:
    """Fine-tune a Wav2Vec2 model on Arabic data using HuggingFace Trainer."""
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(
        model_name,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    model.freeze_feature_encoder()

    train_prepared = prepare_dataset_for_wav2vec(train_dataset, processor)
    eval_prepared = prepare_dataset_for_wav2vec(eval_dataset, processor)

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    training_args = TrainingArguments(
        output_dir=output_dir,
        group_by_length=True,
        per_device_train_batch_size=batch_size,
        evaluation_strategy="epoch",
        num_train_epochs=num_epochs,
        fp16=torch.cuda.is_available(),
        save_steps=400,
        eval_steps=400,
        logging_steps=50,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        save_total_limit=3,
        push_to_hub=False,
        report_to="none",
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_prepared,
        eval_dataset=eval_prepared,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Fine-tuned model saved to {output_dir}")
