# baseline LED-based summarization model.
from transformers import LEDTokenizer, LEDForConditionalGeneration

class LEDSummarizer:
    def __init__(self, model_name: str, device: str):
        self.device = device
        self.tokenizer = LEDTokenizer.from_pretrained(model_name)
        self.model = LEDForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)

    def train_step(self, documents, summaries, max_input_length, max_target_length):
        inputs = self.tokenizer(
            documents,
            truncation=True,
            padding="max_length",
            max_length=max_input_length,
            return_tensors="pt"
        )

        labels = self.tokenizer(
            summaries,
            truncation=True,
            padding="max_length",
            max_length=max_target_length,
            return_tensors="pt"
        )

        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        labels_ids = labels.input_ids.to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels_ids
        )

        loss = outputs.loss
        return loss


    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
