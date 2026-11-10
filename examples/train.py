"""Small Kaggle-ready fine-tuning entry point; see README for data paths."""

import argparse
import random

import torch
from torch.utils.data import DataLoader

from bertdm import DEFAULT_CHECKPOINT, DEFAULT_REVISION, DecisionConfig, DecisionPipeline, load_splits


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dataset", required=True,
                      choices=["cosmos", "wikihop"])
  parser.add_argument("--train-file", required=True)
  parser.add_argument("--validation-file", required=True)
  parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
  parser.add_argument("--revision", default=DEFAULT_REVISION)
  parser.add_argument("--max-length", type=int, default=512)
  parser.add_argument("--head-width", type=int, default=256)
  parser.add_argument("--dropout", type=float, default=0.1)
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--epochs", type=int, default=1)
  parser.add_argument("--lr", type=float, default=2e-5)
  parser.add_argument("--seed", type=int, default=17)
  parser.add_argument("--local-files-only", action="store_true")
  parser.add_argument("--output", default="/kaggle/working/decision")
  args = parser.parse_args()
  random.seed(args.seed)
  torch.manual_seed(args.seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
  config = DecisionConfig(checkpoint=args.checkpoint, revision=args.revision, max_length=args.max_length,
                          head_width=args.head_width, dropout=args.dropout)
  pipeline = DecisionPipeline.from_checkpoint(
      config, local_files_only=args.local_files_only)
  splits = load_splits(args.dataset, train_file=args.train_file,
                       validation_file=args.validation_file)
  # Fail with split/index on any rejection; never silently skip.
  encoded = pipeline.tokenize_splits(splits)
  generator = torch.Generator().manual_seed(args.seed)
  loaders = {split: DataLoader(rows, batch_size=args.batch_size, shuffle=split == "train",
                               generator=generator, collate_fn=pipeline.collate)
             for split, rows in encoded.items()}
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = pipeline.model.to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
  for epoch in range(args.epochs):
    model.train()
    for batch in loaders["train"]:
      batch = {k: v.to(device) for k, v in batch.items()}
      optimizer.zero_grad(set_to_none=True)
      output = model(**batch)
      output.loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
    model.eval()
    correct = total = 0
    with torch.inference_mode():
      for batch in loaders["validation"]:
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)
        correct += (output.logits.argmax(-1) == batch["labels"]).sum().item()
        total += len(batch["labels"])
    print(
        f"epoch={epoch + 1} validation_accuracy={correct / total:.4f} n={total}")
    pipeline.save_pretrained(args.output)


if __name__ == "__main__":
  main()
