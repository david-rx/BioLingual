"""
Calculate retrieval metrics
using the HuggingFace CLAP implementation.

Note: this script may give very slightly different (~0.1%) results from the paper
due to slightly different audio preprocessing. To recreate the exact results, use
CLAP/experiment_scripts/eval_biolingual.sh
"""

from tqdm import tqdm
import torch
from transformers import AutoProcessor, ClapModel
import pandas as pd
from beans.datasets import ClassificationDataset
from torch.utils.data import DataLoader
from CLAP.src.laion_clap.training.train import get_metrics_biolingual

MODEL_IDENTIFIER = "laion/clap-htsat-unfused"
TEST_SET = "beans/test_set.csv"

def compute_tta():
    device = "mps"

    test_df = pd.read_csv(TEST_SET)
    model = ClapModel.from_pretrained(MODEL_IDENTIFIER).to(device)
    processor = AutoProcessor.from_pretrained(MODEL_IDENTIFIER)
    all_captions = test_df["caption"].tolist()

    dataset = ClassificationDataset(
            metadata_path=TEST_SET,
            num_labels=len(all_captions),
            labels=all_captions,
            unknown_label=None,
            sample_rate=48000,
            max_duration=10,
            feature_type="waveform"
    )
    print("made dataset")
    dataloader = DataLoader(
            dataset=dataset,
            batch_size=128,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            persistent_workers=True
    )
    print("made dataloader")

    audio_embeds = []
    text_embeds = []

    for audios, captions in tqdm(dataloader):
        with torch.no_grad():
            x = [s.cpu().numpy() for s in audios]
            inputs = processor(audios=x, text=captions, return_tensors="pt", sampling_rate=48000, padding=True).to(device)
            model_outputs = model(**inputs)
            audio_embeds.extend(model_outputs.audio_embeds.detach().cpu())
            text_embeds.extend(model_outputs.text_embeds.detach().cpu())

    audio_features = torch.stack(audio_embeds)
    print("audio features shape", audio_features.shape)
    text_features = torch.stack(text_embeds)
    with torch.no_grad():
        get_metrics_biolingual(audio_features=audio_features, text_features=text_features, logit_scale_a=model.logit_scale_a.exp().cpu(), captions=all_captions)

if __name__ == "__main__":
    compute_tta()