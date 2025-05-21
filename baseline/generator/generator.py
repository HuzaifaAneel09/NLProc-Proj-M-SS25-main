from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class Generator:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def build_prompt(self, context: str, question: str) -> str:
        prompt = (
            "You are a helpful assistant. Use the context below to answer the question. "
            "If you cannot find the answer in the context, respond with 'I don't know.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )
        return prompt

    def generate_answer(self, context: str, question: str) -> str:
        prompt = self.build_prompt(context, question)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        outputs = self.model.generate(**inputs, max_length=256)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
