import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import os
import math

# ========== НАСТРОЙКИ ==========
config = {
    "block_size": 512,
    "n_embed": 512,
    "n_head": 8,
    "n_layer": 8,
    "dropout": 0.1,
    "temperature": 0.7,
    "top_k": 40,
    "max_new_tokens": 100,
}


# ========== МОДЕЛЬ ==========
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head = n_head
        self.n_embed = n_embed
        self.head_dim = n_embed // n_head

        self.c_attn = nn.Linear(n_embed, 3 * n_embed)
        self.c_proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(config['dropout'])

        self.register_buffer("bias", torch.tril(torch.ones(config["block_size"], config["block_size"]))
                             .view(1, 1, config["block_size"], config["block_size"]))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if hasattr(F, 'scaled_dot_product_attention'):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.dropout(y)
        return y


class MLP(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.c_fc = nn.Linear(n_embed, 4 * n_embed)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embed, n_embed)
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embed)
        self.attn = CausalSelfAttention(n_embed, n_head)
        self.ln_2 = nn.LayerNorm(n_embed)
        self.mlp = MLP(n_embed)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class SmartGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, config["n_embed"])
        self.wpe = nn.Embedding(config["block_size"], config["n_embed"])
        self.drop = nn.Dropout(config['dropout'])

        self.blocks = nn.ModuleList([
            TransformerBlock(config["n_embed"], config["n_head"])
            for _ in range(config["n_layer"])
        ])

        self.ln_f = nn.LayerNorm(config["n_embed"])
        self.lm_head = nn.Linear(config["n_embed"], vocab_size, bias=False)

        self.wte.weight = self.lm_head.weight

    def forward(self, idx):
        B, T = idx.shape
        assert T <= config["block_size"]

        tok_emb = self.wte(idx)
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


# ========== ЗАГРУЗКА ==========
def load_model():
    print("=" * 60)
    print("🤖 ЗАГРУЗКА МОДЕЛИ")
    print("=" * 60)

    # Загружаем словарь
    with open("cache/vocab.json", "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    stoi = vocab_data['stoi']
    itos = {int(k): v for k, v in vocab_data['itos'].items()}
    vocab_size = vocab_data['vocab_size']
    special_tokens = vocab_data.get('special_tokens', {
        '<BOS>': 0, '<EOS>': 1, '<Q>': 2, '<A>': 3, '<PAD>': 4
    })

    print(f"✅ Загружен словарь: {vocab_size} токенов")

    # Загружаем модель
    checkpoint = torch.load("best_model.pth", map_location='cpu')
    model = SmartGPT(vocab_size)

    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        print(f"✅ Модель загружена! Шаг: {checkpoint.get('step', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print(f"✅ Модель загружена!")

    return model, stoi, itos, special_tokens


# ========== ГЕНЕРАЦИЯ ==========
@torch.no_grad()
def generate_response(model, prompt, stoi, itos, special_tokens,
                      max_new_tokens=100, temperature=0.7, top_k=40):
    model.eval()
    device = next(model.parameters()).device

    # Форматируем промпт КАК ПРИ ОБУЧЕНИИ
    prompt = f"<BOS><Q>{prompt}<A>"

    # Токенизация
    context = []
    for ch in prompt:
        if ch in stoi:
            context.append(stoi[ch])
        else:
            context.append(special_tokens.get('<PAD>', 4))

    context = torch.tensor(context, device=device).unsqueeze(0)

    # Генерация
    for _ in range(max_new_tokens):
        if context.size(1) > config["block_size"]:
            context = context[:, -config["block_size"]:]

        logits = model(context)
        logits = logits[0, -1, :] / temperature

        if top_k > 0:
            top_k_val = min(top_k, len(logits))
            top_k_logits, _ = torch.topk(logits, top_k_val)
            indices_to_remove = logits < top_k_logits[..., -1, None]
            logits[indices_to_remove] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)

        if next_token.item() == special_tokens.get('<EOS>', 1):
            break

        # Останавливаемся при длинном ответе
        if len(context[0]) > 50 and next_token.item() == stoi.get('.', 0):
            break

        context = torch.cat([context, next_token.unsqueeze(0)], dim=1)

    # Декодируем
    response = ''
    for t in context[0]:
        token = itos[t.item()]
        if token not in special_tokens:
            response += token

    # Извлекаем ответ после <A>
    if '<A>' in response:
        response = response.split('<A>')[-1].strip()

    # Обрезаем слишком длинные ответы
    if len(response) > 150:
        response = response[:150] + "..."

    return response if response else "Извините, я не могу ответить."


# ========== ЧАТ ==========
def chat():
    print("\n" + "=" * 60)
    print("💬 ПРОСТОЙ ЧАТ")
    print("=" * 60)

    # Загружаем модель
    model, stoi, itos, special_tokens = load_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    print(f"\n💻 Устройство: {device}")
    print(f"📊 Параметров: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "=" * 60)
    print("Команды: /temp [число] - изменить креативность, /quit - выход")
    print("=" * 60)

    temperature = 0.7
    print(f"\n🤖 Бот готов! (креативность={temperature})")
    print("💬 Задайте вопрос...\n")

    while True:
        user_input = input("👤 Вы: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ['/quit', 'quit', 'выход']:
            print("\n🤖 Бот: До свидания! 👋")
            break

        if user_input.startswith('/temp'):
            try:
                temperature = float(user_input.split()[1])
                temperature = max(0.3, min(1.2, temperature))
                print(f"✅ Креативность = {temperature}")
                continue
            except:
                print("❌ Используйте: /temp 0.7")
                continue

        print("🤖 Бот: ", end="", flush=True)
        response = generate_response(model, user_input, stoi, itos, special_tokens,
                                     temperature=temperature)
        print(response)
        print()


if __name__ == "__main__":
    chat()
