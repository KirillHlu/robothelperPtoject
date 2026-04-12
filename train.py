import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import re

block_size = 128
batch_size = 32  # Меньше батч = лучше качество
n_embed = 384  # Больше нейронов
n_head = 6
n_layer = 6
lr = 3e-4
steps = 20000  # Больше шагов
warmup_steps = 2000

print("🧠 УМНОЕ ОБУЧЕНИЕ (4-6 часов)")
print("=" * 50)

# ============ ЗАГРУЗКА БОЛЬШЕ ДАННЫХ ============
print("📖 Загрузка данных...")
with open("dialogs.txt", "r", encoding="utf-8") as f:
    # Берем первые 50 млн символов (10x больше!)
    text = f.read()[:50000000]

print(f"   Символов: {len(text):,}")
print(f"   Диалогов: {text.count('Вопрос:'):,}")

# ============ УЛУЧШЕННЫЙ СЛОВАРЬ ============
# Добавляем все возможные символы русского языка
extra_chars = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ .,!?;:()\"'-"
chars = sorted(list(set(text + extra_chars)))
vocab_size = len(chars)
print(f"   Словарь: {vocab_size} символов")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"   Train: {len(train_data):,} символов")
print(f"   Val: {len(val_data):,} символов")
print()


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


# ============ МОДЕЛЬ ============
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

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
        head_size = n_embed // n_head
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embed, n_embed)
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
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHead()
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, n_embed)
        self.pos_embed = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)
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


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GPT().to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"🧠 Параметров: {total_params:,}")
print(f"🎮 Устройство: {device}")
print()

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)


def get_lr(step):
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (steps - warmup_steps)
    return lr * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)

print("🚀 Старт обучения!")
print("⏱️  Будет готово через 4-6 часов")
print()

start = time.time()

for step in range(steps):
    x, y = get_batch('train')
    x, y = x.to(device), y.to(device)

    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if step % 1000 == 0:
        elapsed = time.time() - start
        eta = (steps - step) * (elapsed / (step + 1)) / 60
        print(
            f"step {step:5d}/{steps} | loss {loss.item():.3f} | lr {scheduler.get_last_lr()[0]:.2e} | ETA {eta:.0f}мин")

        # Сохраняем чекпоинт
        if step > 0 and step % 5000 == 0:
            torch.save(model.state_dict(), f"checkpoint_{step}.pth")

total_time = (time.time() - start) / 60
print(f"\n✅ ГОТОВО! Время: {total_time:.1f} минут")

# Сохраняем финальную модель
torch.save({
    "model": model.state_dict(),
    "stoi": stoi,
    "itos": itos,
    "config": {
        "block_size": block_size,
        "vocab_size": vocab_size,
        "n_embed": n_embed,
        "n_head": n_head,
        "n_layer": n_layer
    }
}, "smart_model.pth")
print("💾 Модель сохранена в smart_model.pth")

# ============ ТЕСТ ============
print("\n" + "=" * 50)
print("💬 ТЕСТ ДИАЛОГА")
print("=" * 50)

model.eval()


def ask(question, max_tokens=100, temperature=0.7):
    prompt = f"Вопрос: {question}\nОтвет: "
    context = torch.tensor([[stoi.get(c, 0) for c in prompt]], device=device)

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(context[:, -block_size:])
            logits = logits[0, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            # Top-k sampling
            top_k = 50
            probs, indices = torch.topk(probs, top_k)
            next_token = indices[torch.multinomial(probs, 1)]

            context = torch.cat([context, next_token.unsqueeze(0)], dim=1)

            # Стоп при переносе строки
            if itos[next_token.item()] == '\n':
                break

    generated = ''.join([itos[int(i)] for i in context[0].tolist()])
    if "Ответ:" in generated:
        response = generated.split("Ответ:")[-1].strip()
    else:
        response = generated[len(prompt):].strip()

    return response


test_questions = [
    "Привет, как дела?",
    "Что посоветуешь?",
    "Как стать счастливым?"
]

for q in test_questions:
    print(f"\n❓ {q}")
    print(f"🤖 {ask(q)}")

print("\n" + "=" * 50)
print("🎉 ГОТОВО!")
print("=" * 50)
