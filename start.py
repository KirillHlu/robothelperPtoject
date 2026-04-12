import torch
import torch.nn as nn
from torch.nn import functional as F

# ============ ТА ЖЕ САМАЯ МОДЕЛЬ (скопируй классы) ============
class Head(nn.Module):
    def __init__(self, head_size, n_embed, block_size):
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
    def __init__(self, n_embed, n_head, block_size):
        super().__init__()
        head_size = n_embed // n_head
        self.heads = nn.ModuleList([Head(head_size, n_embed, block_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
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
    def __init__(self, n_embed, n_head, block_size):
        super().__init__()
        self.sa = MultiHead(n_embed, n_head, block_size)
        self.ff = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(vocab_size, config['n_embed'])
        self.pos_embed = nn.Embedding(config['block_size'], config['n_embed'])
        self.blocks = nn.Sequential(*[Block(config['n_embed'], config['n_head'], config['block_size']) 
                                       for _ in range(config['n_layer'])])
        self.ln = nn.LayerNorm(config['n_embed'])
        self.head = nn.Linear(config['n_embed'], vocab_size)

    def forward(self, x):
        B, T = x.shape
        tok = self.token_embed(x)
        pos = self.pos_embed(torch.arange(T, device=x.device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln(x)
        return self.head(x)

# ============ ЗАГРУЗКА МОДЕЛИ ============
print("Загрузка модели...")

# Загружаем сохраненную модель
checkpoint = torch.load("comfortable_gpt.pth", map_location='cpu')

config = checkpoint['config']
stoi = checkpoint['stoi']
itos = checkpoint['itos']
vocab_size = len(stoi)

# Создаем модель
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GPT(vocab_size, config).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

print(f"Модель загружена! Устройство: {device}")
print()

# ============ ФУНКЦИЯ ДЛЯ ОБЩЕНИЯ ============
def chat(question, max_tokens=100, temperature=0.7):
    prompt = f"Вопрос: {question}\nОтвет: "
    context = torch.tensor([[stoi.get(c, 0) for c in prompt]], device=device)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(context[:, -config['block_size']:])
            logits = logits[0, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            context = torch.cat([context, next_token.unsqueeze(0)], dim=1)
            
            if itos[next_token.item()] == '\n':
                break
    
    generated = ''.join([itos[int(i)] for i in context[0].tolist()])
    if "Ответ:" in generated:
        return generated.split("Ответ:")[-1].strip()
    return generated[len(prompt):].strip()

# ============ ЧАТ ============
print("=" * 50)
print("ЧАТ ЗАПУЩЕН")
print("Введите 'выход' или 'quit' для выхода")
print("=" * 50)

while True:
    user_input = input("\nВы: ").strip()
    
    if user_input.lower() in ['выход', 'quit', 'exit']:
        print("До свидания!")
        break
    
    if user_input:
        response = chat(user_input)
        print(f"Бот: {response}")
