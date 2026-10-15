"""Apoio mecânico ao Lab 2. As decisões e implementações avaliadas ficam no notebook."""
from pathlib import Path
import hashlib
import random
import numpy as np
import pandas as pd
import torch
from torch import nn

FEATURES = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol',
]
SHA256_ORIGINAL = '4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e'


def fixar_semente(semente=42):
    """Fixa geradores usuais; não promete igualdade entre hardwares ou versões."""
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semente)


def carregar_base(caminho=None):
    """Lê o CSV original e acrescenta alvo e controles; não ajusta transformações."""
    caminho = Path(caminho) if caminho else Path(__file__).parent/'dados/winequality-red.csv'
    if hashlib.sha256(caminho.read_bytes()).hexdigest() != SHA256_ORIGINAL:
        raise ValueError('O CSV foi alterado. Use o original do pacote ou documente outra fonte explicitamente.')
    df = pd.read_csv(caminho, sep=';')
    if list(df.columns) != FEATURES + ['quality'] or len(df) != 1599:
        raise ValueError('Esquema ou tamanho inesperado para o arquivo original.')
    if df.isna().any().any():
        raise ValueError('O arquivo original não possui valores ausentes.')
    df['alvo'] = (df['quality'] >= 7).astype('int64')
    df['id_linha'] = np.arange(len(df))
    # Mesmo perfil numérico recebe o mesmo grupo. Não é uma identidade física.
    df['grupo_perfil'] = pd.util.hash_pandas_object(df[FEATURES], index=False).astype(str)
    return df


def criar_modelo(n_entradas=11, batchnorm=False):
    """Rede de referência. Saída [B] de logits, compatível com rótulos [B]."""
    camadas = [nn.Linear(n_entradas, 16)]
    if batchnorm:
        camadas.append(nn.BatchNorm1d(16))
    camadas += [nn.ReLU(), nn.Linear(16, 1), _AchatarSaida()]
    return nn.Sequential(*camadas)


class _AchatarSaida(nn.Module):
    def forward(self, x):
        return x.squeeze(-1)


def _lote(x, y, dispositivo):
    x = x.to(dispositivo, dtype=torch.float32)
    y = y.to(dispositivo, dtype=torch.float32)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError('Esperado X [B, características] e y [B], alinhados.')
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('Lote com NaN ou infinito.')
    if not ((y == 0) | (y == 1)).all():
        raise ValueError('Os rótulos devem ser 0 ou 1.')
    return x, y


def treinar_epoca(modelo, loader, otimizador, criterio, dispositivo='cpu'):
    """Treina uma época. Critério deve usar reduction='mean'. Retorna perda média."""
    modelo.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = _lote(x, y, dispositivo)
        otimizador.zero_grad(set_to_none=True)
        logits = modelo(x)
        if logits.shape != y.shape:
            raise ValueError('Saída e alvo precisam ter a mesma forma.')
        perda = criterio(logits, y)
        perda.backward()
        otimizador.step()
        total += perda.item() * len(y)
        n += len(y)
    if n == 0:
        raise ValueError('Carregador vazio: confira batch_size e drop_last.')
    return total / n


@torch.no_grad()
def avaliar_perda(modelo, loader, criterio, dispositivo='cpu'):
    """Avalia uma partição sem gradientes e sem descartar exemplos."""
    if loader.drop_last:
        raise ValueError('Na avaliação, use drop_last=False.')
    modelo.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = _lote(x, y, dispositivo)
        logits = modelo(x)
        if logits.shape != y.shape:
            raise ValueError('Saída e alvo precisam ter a mesma forma.')
        total += criterio(logits, y).item() * len(y)
        n += len(y)
    if not n:
        raise ValueError('Partição vazia.')
    return total / n


@torch.no_grad()
def obter_escores(modelo, loader, dispositivo='cpu'):
    """Retorna (rótulos, escores) juntos e alinhados, na ordem percorrida."""
    if loader.drop_last:
        raise ValueError('Não descarte o último lote da avaliação.')
    modelo.eval()
    rotulos, escores = [], []
    for x, y in loader:
        x, y = _lote(x, y, dispositivo)
        logits = modelo(x)
        if logits.shape != y.shape:
            raise ValueError('Saída e alvo precisam ter a mesma forma.')
        rotulos.append(y.cpu().numpy())
        escores.append(torch.sigmoid(logits).cpu().numpy())
    if not rotulos:
        raise ValueError('Partição vazia.')
    return np.concatenate(rotulos).astype(int), np.concatenate(escores)
