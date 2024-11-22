from datasets import load_dataset, load_from_disk
import sys

# dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
# text_data = " ".join(dataset["text"])
# dataset.save_to_disk("./wikitext-103-v1-train")

dataset = load_dataset("bookcorpus", trust_remote_code=True)
dataset.save_to_disk("./bookcorpus")

print("dataset saved")