# clip_scratch.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision
import random
import time

# ----------------------------
# Utilities
# ----------------------------
def l2norm(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

# ----------------------------
# Patch embedding for ViT (simple)
# ----------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=512):
        super().__init__()
        assert img_size % patch_size == 0
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

    def forward(self, x):
        # x: B,C,H,W
        x = self.proj(x)                 # B, embed_dim, H/ps, W/ps
        x = x.flatten(2).transpose(1, 2) # B, N, embed_dim
        return x

# ----------------------------
# Simple Transformer Encoder Block
# ----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim*mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        # x: B, N, D
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        y = self.norm2(x)
        y = self.mlp(y)
        x = x + y
        return x

# ----------------------------
# ViT encoder (small)
# ----------------------------
class ViTEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=512, depth=6, num_heads=8, mlp_ratio=4.):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        # positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.0)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # B, N, D
        cls = self.cls_token.expand(B, -1, -1) # B,1,D
        x = torch.cat([cls, x], dim=1) # B, N+1, D
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # return CLS embedding
        return x[:,0]

# ----------------------------
# Simple text tokenizer & dataset helper (very small)
# ----------------------------
class SimpleTokenizer:
    def __init__(self, max_len=32):
        self.max_len = max_len
        self.vocab = {"<pad>":0, "<unk>":1, "<bos>":2, "<eos>":3}
        self.inv_vocab = {0:"<pad>",1:"<unk>",2:"<bos>",3:"<eos>"}
        self.vocab_size = len(self.vocab)

    def build_vocab(self, texts, min_freq=1):
        freq = {}
        for t in texts:
            for w in t.strip().lower().split():
                freq[w] = freq.get(w,0)+1
        for w,c in freq.items():
            if c >= min_freq and w not in self.vocab:
                idx = len(self.vocab)
                self.vocab[w] = idx
                self.inv_vocab[idx] = w
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        toks = text.strip().lower().split()
        ids = [self.vocab.get(w, self.vocab["<unk>"]) for w in toks]
        ids = [self.vocab["<bos>"]] + ids + [self.vocab["<eos>"]]
        if len(ids) < self.max_len:
            ids = ids + [self.vocab["<pad>"]] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
            if ids[-1] != self.vocab["<eos>"]:
                ids[-1] = self.vocab["<eos>"]
        return torch.tensor(ids, dtype=torch.long)

# ----------------------------
# Text encoder (embedding + positional + transformer stack)
# ----------------------------
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, max_len=32, embed_dim=512, depth=6, num_heads=8, mlp_ratio=4.):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.drop = nn.Dropout(0.0)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.max_len = max_len
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, token_ids):
        # token_ids: B, L
        x = self.token_emb(token_ids) + self.pos_emb[:, :token_ids.shape[1], :]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # return pooled representation: take the <bos> (index 0 if we encoded that way) or mean over non-pad
        # We'll assume <bos> at position 0
        return x[:,0]

# ----------------------------
# Projection head and CLIP model wrapper
# ----------------------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x)

class CLIPModel(nn.Module):
    def __init__(self, image_encoder, text_encoder, embed_dim=256):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.img_proj = ProjectionHead(image_encoder.patch_embed.embed_dim if hasattr(image_encoder.patch_embed, "embed_dim") else image_encoder.cls_token.shape[-1], embed_dim)
        self.txt_proj = ProjectionHead(text_encoder.token_emb.embedding_dim, embed_dim)
        # logit scale (learnable)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1/0.07))

    def forward(self, images, token_ids):
        img_feats = self.image_encoder(images)    # B, D1
        txt_feats = self.text_encoder(token_ids)  # B, D2
        img_emb = self.img_proj(img_feats)        # B, d
        txt_emb = self.txt_proj(txt_feats)        # B, d

        img_emb = l2norm(img_emb)
        txt_emb = l2norm(txt_emb)
        # similarities
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * (img_emb @ txt_emb.t())  # B, B
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text

# ----------------------------
# InfoNCE contrastive loss (symmetric)
# ----------------------------
def clip_loss(logits_per_image):
    # logits_per_image: B x B (image-to-text)
    # target is diagonal
    B = logits_per_image.shape[0]
    labels = torch.arange(B, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_image.t(), labels)
    return (loss_i + loss_t) / 2

# ----------------------------
# Demo dataset: CIFAR10 with synthetic captions
# Replace this with your real image-text dataset loader
# ----------------------------
class CIFAR10WithCaptions(Dataset):
    def __init__(self, root="./data", train=True, transform=None, tokenizer=None):
        self.ds = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
        self.labels = self.ds.classes
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        caption = f"a photo of a {self.labels[label]}"
        token_ids = self.tokenizer.encode(caption)
        return img, token_ids

# ----------------------------
# Training loop (demo)
# ----------------------------
def train_clip_demo(epochs=5, batch_size=256, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # transforms (CIFAR sized -> we'll resize to 224 to use ViT patch 16)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
    ])

    # build dataset + tokenizer
    tokenizer = SimpleTokenizer(max_len=16)
    # build vocab from CIFAR label captions (small)
    sample_texts = [f"a photo of a {c}" for c in datasets.CIFAR10(root="./data", train=True, download=True).classes]
    tokenizer.build_vocab(sample_texts)
    train_ds = CIFAR10WithCaptions(transform=transform, tokenizer=tokenizer, train=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # encoders
    image_encoder = ViTEncoder(img_size=224, patch_size=16, embed_dim=512, depth=6, num_heads=8)
    text_encoder = TextEncoder(vocab_size=tokenizer.vocab_size, max_len=16, embed_dim=512, depth=6, num_heads=8)

    model = CLIPModel(image_encoder, text_encoder, embed_dim=256).to(device)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith("cuda")))

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for images, token_ids in train_loader:
            images = images.to(device)
            token_ids = token_ids.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.startswith("cuda"))):
                logits_i2t, logits_t2i = model(images, token_ids)
                loss = clip_loss(logits_i2t)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch} | loss {avg_loss:.4f} | time {time.time()-t0:.1f}s")

    # Save example model
    torch.save(model.state_dict(), "clip_scratch.pth")
    print("Saved clip_scratch.pth")

if __name__ == "__main__":
    train_clip_demo(epochs=3, batch_size=256)
