import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

max_seq_length = 100
total_word_num = 100

# Device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SummarizationDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, max_len):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_text = self.targets[idx]

        input_encodings = self.tokenizer(input_text, truncation=True, padding='max_length', max_length=self.max_len)
        target_encodings = self.tokenizer(target_text, truncation=True, padding='max_length', max_length=self.max_len)

        input_ids = torch.tensor(input_encodings['input_ids'])
        attention_mask = torch.tensor(input_encodings['attention_mask'])
        labels = torch.tensor(target_encodings['input_ids'])

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

class TransformerSummarizer(nn.Module):
    def __init__(self, tokenizer, model_name='t5-small'):
        super(TransformerSummarizer, self).__init__()
        self.tokenizer = tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss, outputs.logits

def train_model(model, dataloader, optimizer, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()
            optimizer.step()

def summarize_text(model, tokenizer, text, max_len=512):
    model.eval()
    input_encodings = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=max_len)
    input_ids = input_encodings['input_ids'].to(device)
    attention_mask = input_encodings['attention_mask'].to(device)

    summary_ids = model.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=150, num_beams=2, length_penalty=2.0)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    model = TransformerSummarizer(tokenizer).to(device)

    # 1. 모델 훈련 시 발생하는 오류 파악
    try:
        # Dummy data
        inputs = ["The quick brown fox jumps over the lazy dog."] * 100
        targets = ["A fox jumps over a dog."] * 100

        dataset = SummarizationDataset(inputs, targets, tokenizer, max_len=512)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        # 2. 모델 훈련
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        train_model(model, dataloader, optimizer, num_epochs=3)

        # 3. 요약 결과 출력
        text = "The quick brown fox jumps over the lazy dog."
        summary = summarize_text(model, tokenizer, text)
        print(f"Summary: {summary}")

    except Exception as e:
        print(f"Error occurred during training: {e}")
