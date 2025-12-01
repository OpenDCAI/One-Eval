from huggingface_hub import HfApi, DatasetCard

# api = HfApi()
card = DatasetCard.load("openai/gsm8k")   # dataset id
text = card.text

print(card)
# print(text[:1000])   # README 前 1000 字
