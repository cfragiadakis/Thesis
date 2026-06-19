#!/usr/bin/env python
"""Extract frozen CLIP ViT frame embeddings for the DIPSER dataset."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


DEFAULT_DATASET_PATH = Path("Data/dipser_dataset.parquet")
DEFAULT_CLIP_MODEL_NAME = "ViT-L/14"
DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_WORKERS = 8
DEFAULT_FEATURE_DTYPE = np.float32
DEFAULT_NORMALIZE_FEATURES = True


class FeatureImageDataset(Dataset):
    def __init__(self, frame_df, transform, image_size):
        self.frame_df = frame_df.reset_index(drop=True)
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.frame_df)

    def __getitem__(self, idx):
        row = self.frame_df.iloc[idx]
        image_path = row["image_path"]

        try:
            image = Image.open(image_path).convert("RGB")
            failed = False
        except Exception:
            image = Image.new("RGB", (self.image_size, self.image_size))
            failed = True

        image = self.transform(image)
        return image, int(row["feature_row"]), row["image_path"], failed


class CLIPFeatureExtractor(nn.Module):
    def __init__(self, model, normalize=True):
        super().__init__()
        self.model = model
        self.normalize = normalize

    def forward(self, images):
        features = self.model.encode_image(images)
        features = features.float()
        if self.normalize:
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return features


def build_frame_dataframe(dataset_path):
    df = pd.read_parquet(dataset_path, engine="pyarrow")
    frame_df = (
        df[["subject_experiment_id", "subject_id", "time_sec", "image_path"]]
        .dropna(subset=["image_path"])
        .drop_duplicates(subset=["image_path"])
        .reset_index(drop=True)
    )
    frame_df["feature_row"] = np.arange(len(frame_df), dtype=np.int64)
    return df, frame_df


def build_feature_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )


def build_loader(frame_df, image_size, batch_size, num_workers):
    dataset = FeatureImageDataset(frame_df, build_feature_transform(image_size), image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return loader


def load_feature_extractor(model_name, device, normalize):
    try:
        import clip
    except ImportError as exc:
        raise ImportError(
            "The OpenAI CLIP package is required for feature extraction. "
            "Install it in the Snellius environment with: "
            "pip install git+https://github.com/openai/CLIP.git"
        ) from exc

    clip_model, _ = clip.load(model_name, device=device, jit=False)
    clip_model.eval()

    for parameter in clip_model.parameters():
        parameter.requires_grad = False

    feature_extractor = CLIPFeatureExtractor(clip_model, normalize=normalize).to(device)
    feature_extractor.eval()
    return feature_extractor


def extract_features(
    frame_df,
    loader,
    feature_extractor,
    features_path,
    feature_dtype,
    feature_dim,
    model_name,
    device,
):
    features = np.lib.format.open_memmap(
        features_path,
        mode="w+",
        dtype=feature_dtype,
        shape=(len(frame_df), feature_dim),
    )

    failed_images = []

    with torch.inference_mode():
        progress = tqdm(loader, desc=f"Extracting CLIP {model_name} features")
        for images, feature_rows, image_paths, failed in progress:
            images = images.to(device, non_blocking=True)
            batch_features = feature_extractor(images)
            batch_features = batch_features.cpu().numpy().astype(feature_dtype, copy=False)

            features[feature_rows.numpy()] = batch_features

            for path, did_fail in zip(image_paths, failed.numpy()):
                if bool(did_fail):
                    failed_images.append(path)

    features.flush()
    return failed_images


def write_outputs(
    frame_df,
    failed_images,
    dataset_path,
    features_path,
    index_path,
    failed_path,
    meta_path,
    feature_dim,
    feature_dtype,
    model_name,
    safe_name,
    image_size,
    normalize_features,
):
    frame_df.to_parquet(index_path, index=False)
    pd.DataFrame({"image_path": failed_images}).to_csv(failed_path, index=False)

    metadata = {
        "dataset_path": str(dataset_path),
        "features_path": str(features_path),
        "index_path": str(index_path),
        "failed_path": str(failed_path),
        "num_frames": int(len(frame_df)),
        "feature_dim": int(feature_dim),
        "feature_dtype": str(np.dtype(feature_dtype)),
        "model_name": model_name,
        "model_family": "OpenAI CLIP",
        "safe_name": safe_name,
        "image_size": int(image_size),
        "preprocessing": (
            f"Resize({image_size}, bicubic) + CenterCrop({image_size}) "
            "+ CLIP normalize, no augmentation"
        ),
        "feature_token": "encode_image output",
        "features_l2_normalized": bool(normalize_features),
        "failed_images": int(len(failed_images)),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--clip-model-name", default=DEFAULT_CLIP_MODEL_NAME)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.clip_model_name
    safe_name = model_name.lower().replace("/", "").replace("-", "")
    output_dir = args.output_dir or Path(f"Data/ALL_clip_{safe_name}_features")
    output_dir.mkdir(parents=True, exist_ok=True)

    features_path = output_dir / f"ALL_clip_{safe_name}_frame_features.npy"
    index_path = output_dir / f"ALL_clip_{safe_name}_frame_index.parquet"
    failed_path = output_dir / f"ALL_clip_{safe_name}_failed_images.csv"
    meta_path = output_dir / f"ALL_clip_{safe_name}_feature_metadata.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df, frame_df = build_frame_dataframe(args.dataset_path)
    print("Original rows:", len(df))
    print("Unique image paths to encode:", len(frame_df))

    loader = build_loader(frame_df, args.image_size, args.batch_size, args.num_workers)
    feature_extractor = load_feature_extractor(
        model_name,
        device,
        DEFAULT_NORMALIZE_FEATURES,
    )

    with torch.inference_mode():
        dummy = torch.zeros(1, 3, args.image_size, args.image_size, device=device)
        feature_dim = int(feature_extractor(dummy).shape[-1])

    failed_images = extract_features(
        frame_df=frame_df,
        loader=loader,
        feature_extractor=feature_extractor,
        features_path=features_path,
        feature_dtype=DEFAULT_FEATURE_DTYPE,
        feature_dim=feature_dim,
        model_name=model_name,
        device=device,
    )

    metadata = write_outputs(
        frame_df=frame_df,
        failed_images=failed_images,
        dataset_path=args.dataset_path,
        features_path=features_path,
        index_path=index_path,
        failed_path=failed_path,
        meta_path=meta_path,
        feature_dim=feature_dim,
        feature_dtype=DEFAULT_FEATURE_DTYPE,
        model_name=model_name,
        safe_name=safe_name,
        image_size=args.image_size,
        normalize_features=DEFAULT_NORMALIZE_FEATURES,
    )

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
