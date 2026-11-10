"""python examples/infer.py --bundle /kaggle/input/my-model/decision"""

import argparse

import torch

from bertdm import DEFAULT_REVISION, DecisionConfig, DecisionPipeline


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--bundle", help="Saved bundle; omit to initialize a new, untrained scoring head")
  parser.add_argument("--checkpoint", default="chandar-lab/NeoBERT",
                      help="Hub ID or staged local snapshot")
  parser.add_argument("--revision", default=DEFAULT_REVISION,
                      help="Pinned checkpoint commit hash")
  parser.add_argument(
      "--device", default="cuda" if torch.cuda.is_available() else "cpu")
  args = parser.parse_args()
  pipeline = (DecisionPipeline.from_pretrained(args.bundle, device=args.device) if args.bundle
              else DecisionPipeline.from_checkpoint(DecisionConfig(checkpoint=args.checkpoint, revision=args.revision)))
  pipeline.model.to(args.device)
  if not args.bundle:
    print("Scoring head is newly initialized; fine-tune before using predictions.")
  result = pipeline.predict([{
      "context": "Mira left her umbrella at home. It rained on her walk to school.",
      "question": "Why did Mira arrive wet?",
      "choices": ["She walked in the rain without an umbrella.", "The sun dried her coat."],
      "label": None,
  }], return_pooled=True)
  print("Choice probabilities:", result["probabilities"].cpu().tolist())
  print("Predicted zero-based labels:", result["predictions"].cpu().tolist())


if __name__ == "__main__":
  main()
