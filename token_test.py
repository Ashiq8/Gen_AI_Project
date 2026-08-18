import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4o-mini")

sentence = {
    "English": "Artificial Intelligence is transforming the world.",
    "Hindi": "कृत्रिम बुद्धिमत्ता दुनिया को बदल रही है।",
    "Code": "def add(a,b): return a+b"
}

token_counts = {}

for lang, text in sentence.items():
    tokens = encoding.encode(text)
    token_counts[lang] = len(tokens)

    print(f"\n{lang} Sentence: {text}")
    print("Tokens:", tokens)
    print("Tokens Count:", len(tokens))
