import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import gzip
import json
import os

config = {
    "block_size": 128,
    "batch_size": 64,
    "n_embed": 384,
    "n_head": 6,
    "n_layer": 6,
    "lr": 3e-4,
    "steps": 20000,
    "warmup_steps": 2000,
    "eval_interval": 500,
    "eval_iters": 100,
    "temperature": 0.7,
    "top_k": 50
}

print("=" * 60)
print("КОМФОРТНАЯ НЕЙРОСЕТЬ ДЛЯ ОБЩЕНИЯ")
print("=" * 60)

print("\nЗагрузка диалогов...")
dialogues = []
with gzip.open("conversations.jsonl.gz", "rt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 100000:
            break
        data = json.loads(line)
        conversation = data['conversation']
        lines = conversation.strip().split('\n')
        cleaned = []
        for l in lines:
            l = l.strip()
            if l.startswith('—') or l.startswith('-') or l.startswith('•'):
                l = l[1:].strip()
            if l and len(l) > 2:
                cleaned.append(l)
        for j in range(0, len(cleaned) - 1, 2):
            if j + 1 < len(cleaned):
                dialogues.append(f"Вопрос: {cleaned[j]}\nОтвет: {cleaned[j+1]}\n\n")
        if (i + 1) % 20000 == 0:
            print(f"   Загружено {i + 1} диалогов...")

text = "".join(dialogues)
print(f"\nЗагружено {len(dialogues)} диалогов")
print(f"Всего символов: {len(text):,}")

print("\nСоздание словаря...")
russian_chars = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
punctuation = " .,!?;:()\"'-—\n"
all_chars = sorted(list(set(text + russian_chars + punctuation)))
vocab_size = len(all_chars)

stoi = {ch: i for i, ch in enumerate(all_chars)}
itos = {i: ch for i, ch in enumerate(all_chars)}

data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Словарь: {vocab_size} символов")
print(f"Train: {len(train_data):,} символов")
print(f"Val: {len(val_data):,} символов")

del dialogues
del text

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config["block_size"] - 1, (config["batch_size"],))
    x = torch.stack([data[i:i+config["block_size"]] for i in ix])
    y = torch.stack([data[i+1:i+config["block_size"]+1] for i in ix])
    return x, y

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config["n_embed"], head_size, bias=False)
        self.query = nn.Linear(config["n_embed"], head_size, bias=False)
        self.value = nn.Linear(config["n_embed"], head_size, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.register_buffer("tril", torch.tril(torch.ones(config["block_size"], config["block_size"])))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = config["n_embed"] // config["n_head"]
        self.heads = nn.ModuleList([Head(head_size) for _ in range(config["n_head"])])
        self.proj = nn.Linear(config["n_embed"], config["n_embed"])
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config["n_embed"], 4 * config["n_embed"]),
            nn.GELU(),
            nn.Linear(4 * config["n_embed"], config["n_embed"]),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHead()
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(config["n_embed"])
        self.ln2 = nn.LayerNorm(config["n_embed"])

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class ComfortableGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, config["n_embed"])
        self.pos_embed = nn.Embedding(config["block_size"], config["n_embed"])
        self.blocks = nn.Sequential(*[Block() for _ in range(config["n_layer"])])
        self.ln = nn.LayerNorm(config["n_embed"])
        self.head = nn.Linear(config["n_embed"], vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.shape
        tok = self.token_embed(x)
        pos = self.pos_embed(torch.arange(T, device=x.device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln(x)
        return self.head(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ComfortableGPT().to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nПараметров: {total_params:,}")
print(f"Устройство: {device}")
print()

optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)

def get_lr(step):
    if step < config["warmup_steps"]:
        return config["lr"] * (step + 1) / config["warmup_steps"]
    progress = (step - config["warmup_steps"]) / (config["steps"] - config["warmup_steps"])
    return config["lr"] * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config["eval_iters"])
        for k in range(config["eval_iters"]):
            x, y = get_batch(split)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            losses[k] = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        out[split] = losses.mean()
    model.train()
    return out

@torch.no_grad()
def generate_response(question, max_new_tokens=120):
    prompt = f"Вопрос: {question}\nОтвет: "
    context = torch.tensor([[stoi.get(c, 0) for c in prompt]], device=device)
    
    for _ in range(max_new_tokens):
        logits = model(context[:, -config["block_size"]:])
        logits = logits[0, -1, :] / config["temperature"]
        
        top_k = min(config["top_k"], vocab_size)
        top_logits, top_indices = torch.topk(logits, top_k)
        probs = F.softmax(top_logits, dim=-1)
        next_token = top_indices[torch.multinomial(probs, 1)]
        
        context = torch.cat([context, next_token.unsqueeze(0)], dim=1)
        
        if itos[next_token.item()] == '\n':
            break
    
    generated = ''.join([itos[int(i)] for i in context[0].tolist()])
    if "Ответ:" in generated:
        return generated.split("Ответ:")[-1].strip()
    return generated[len(prompt):].strip()

os.makedirs("checkpoints", exist_ok=True)

print("Начало обучения!")
print(f"Всего шагов: {config['steps']}")
print()

start_time = time.time()
best_val_loss = float('inf')

for step in range(config["steps"]):
    x, y = get_batch('train')
    x, y = x.to(device), y.to(device)
    
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    
    if step % config["eval_interval"] == 0 and step > 0:
        elapsed = time.time() - start_time
        eta = (config["steps"] - step) * (elapsed / (step + 1)) / 60
        
        losses = estimate_loss()
        
        print(f"\nstep {step:6d}/{config['steps']}")
        print(f"   train_loss: {losses['train']:.4f}")
        print(f"   val_loss: {losses['val']:.4f}")
        print(f"   lr: {scheduler.get_last_lr()[0]:.2e}")
        print(f"   ETA: {eta:.0f} мин")
        
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "stoi": stoi,
                "itos": itos,
                "val_loss": losses['val'],
                "config": config
            }, f"checkpoints/best_model_step_{step}_loss_{losses['val']:.4f}.pth")
            print(f"   ✨ Новая лучшая модель! Loss: {losses['val']:.4f}")

print("\n" + "=" * 60)
print("Сохранение финальной модели")
print("=" * 60)

torch.save({
    "model": model.state_dict(),
    "stoi": stoi,
    "itos": itos,
    "config": config
}, "comfortable_gpt.pth")

print("✅ Модель сохранена в comfortable_gpt.pth")

print("\n" + "=" * 60)
print("ТЕСТ ОБЩЕНИЯ")
print("=" * 60)

model.eval()

test_questions = [
    "Привет, как дела?",
    "Что посоветуешь?",
    "Как стать счастливым?",
    "Расскажи шутку",
    "Что ты умеешь?"
]

for q in test_questions:
    response = generate_response(q)
    print(f"\n❓ {q}")
    print(f"🤖 {response}")

print("\n" + "=" * 60)
print("ЧАТ-БОТ ЗАПУЩЕН")
print("Введите 'выход' или 'quit' для выхода")
print("=" * 60)

while True:
    user_input = input("\nВы: ").strip()
    if user_input.lower() in ['выход', 'quit', 'exit']:
        print("До свидания!")
        break
    if user_input:
        response = generate_response(user_input)
        print(f"Бот: {response}")
