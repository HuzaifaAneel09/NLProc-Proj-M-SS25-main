# pipeline.py
from datetime import datetime
from baseline.generator.generator import Generator
from baseline.retriever.retriever import Retriever

class RAGPipeline:
    def __init__(self, data_path: str, retriever_store: str, group_id: str = "Neural Nets"):
        self.retriever = Retriever()
        self.generator = Generator()
        self.retriever.initialize_or_load(data_path, retriever_store)
        self.group_id = group_id

    def run(self, question: str, top_k: int = 3) -> dict:
        retrieved_chunks = self.retriever.query(question, top_k)
        context = "\n\n".join(retrieved_chunks)
        prompt = self.generator.build_prompt(context, question)
        answer = self.generator.generate_answer(context, question)

        return {
            "question": question,
            "retrieved_chunks": retrieved_chunks,
            "prompt": prompt,
            "generated_answer": answer,
            "timestamp": datetime.utcnow().isoformat(),
            "group_id": self.group_id
        }
