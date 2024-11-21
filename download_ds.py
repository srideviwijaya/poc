from datasets import load_dataset, load_from_disk
import sys

dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
dataset = load_from_disk(data_dir)
text_data = " ".join(dataset["text"])
dataset.save_to_disk("./wikitext-103-v1-train")

print("dataset saved")
sys.exit(0)