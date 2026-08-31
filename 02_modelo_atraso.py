from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "outputs" / "base_pedidos_modelagem.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "modelo_atraso"
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


def criar_features(df):
    """Cria somente informações que podem existir no momento da compra."""

    df = df.copy()

    # Converter datas
    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"],
        errors="coerce",
    )

    # Variáveis de calendário conhecidas no checkout
    df["purchase_month_number"] = df["order_purchase_timestamp"].dt.month
    df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour

    # Se cliente e vendedor estão no mesmo estado
    df["same_state"] = (
        df["customer_state"].fillna("unknown")
        == df["seller_state"].fillna("unknown")
    ).astype(int)

    # Evitar divisão por zero
    df["freight_to_price_ratio"] = (
        df["frete_total"] / df["preco_total"].replace(0, np.nan)
    )

    return df


print("=" * 70)
print("OLIST | 02 - MODELO DE PREVISAO DE ATRASO")
print("=" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Não encontrei {INPUT_FILE}. "
        "Execute primeiro o arquivo 01_exploracao.py."
    )

df = pd.read_csv(INPUT_FILE, low_memory=False)

# Converter datas para escolher pedidos entregues e criar o target
df["order_purchase_timestamp"] = pd.to_datetime(
    df["order_purchase_timestamp"],
    errors="coerce",
)

df["order_delivered_customer_date"] = pd.to_datetime(
    df["order_delivered_customer_date"],
    errors="coerce",
)

df["order_estimated_delivery_date"] = pd.to_datetime(
    df["order_estimated_delivery_date"],
    errors="coerce",
)

# Target: atraso real
df = df[df["order_delivered_customer_date"].notna()].copy()

df["delayed"] = (
    df["order_delivered_customer_date"]
    > df["order_estimated_delivery_date"]
).astype(int)

# Remover registros sem data de compra
df = df[df["order_purchase_timestamp"].notna()].copy()

# Criar apenas features permitidas no momento da compra
df = criar_features(df)

# IMPORTANTE:
# Não usamos delivery date, estimated delivery date, approval time
# ou posting time como feature. Elas só são conhecidas depois do checkout.
features_numericas = [
    "preco_total",
    "frete_total",
    "itens",
    "product_weight_g",
    "purchase_month_number",
    "purchase_dayofweek",
    "purchase_hour",
    "same_state",
    "freight_to_price_ratio",
]

features_categoricas = [
    "customer_state",
    "seller_state",
    "product_category_name",
]

features = features_numericas + features_categoricas
target = "delayed"

# Divisão temporal: últimos 20% dos pedidos são teste
df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)

ponto_corte = int(len(df) * 0.80)
treino = df.iloc[:ponto_corte].copy()
teste = df.iloc[ponto_corte:].copy()

X_train = treino[features]
y_train = treino[target]

X_test = teste[features]
y_test = teste[target]

print(f"\nPedidos para treino: {len(treino):,}")
print(f"Pedidos para teste: {len(teste):,}")
print(
    "Período de treino: "
    f"{treino['order_purchase_timestamp'].min().date()} até "
    f"{treino['order_purchase_timestamp'].max().date()}"
)
print(
    "Período de teste: "
    f"{teste['order_purchase_timestamp'].min().date()} até "
    f"{teste['order_purchase_timestamp'].max().date()}"
)
print(f"Taxa de atraso no treino: {y_train.mean():.2%}")
print(f"Taxa de atraso no teste: {y_test.mean():.2%}")

# Tratamento dos dados
transformador_numerico = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

transformador_categorico = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessador = ColumnTransformer(
    transformers=[
        ("numericas", transformador_numerico, features_numericas),
        ("categoricas", transformador_categorico, features_categoricas),
    ]
)

# Random Forest funciona bem para primeira versão e permite medir importância
from sklearn.ensemble import RandomForestClassifier

modelo = Pipeline(
    steps=[
        ("preprocessador", preprocessador),
        (
            "modelo",
            RandomForestClassifier(
                n_estimators=250,
                max_depth=14,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

print("\nTreinando modelo...")
modelo.fit(X_train, y_train)

# Probabilidade de atraso
prob_atraso = modelo.predict_proba(X_test)[:, 1]

# Threshold inicial: 0.50
threshold = 0.50
previsao = (prob_atraso >= threshold).astype(int)

# Métricas do modelo
auc = roc_auc_score(y_test, prob_atraso)
precision = precision_score(y_test, previsao, zero_division=0)
recall = recall_score(y_test, previsao, zero_division=0)
f1 = f1_score(y_test, previsao, zero_division=0)

# Baseline: prever que todos os pedidos serão "no prazo"
baseline_pred = np.zeros(len(y_test), dtype=int)
baseline_precision = precision_score(
    y_test,
    baseline_pred,
    zero_division=0,
)
baseline_recall = recall_score(
    y_test,
    baseline_pred,
    zero_division=0,
)

print("\n" + "-" * 70)
print("RESULTADOS DO MODELO")
print("-" * 70)
print(f"AUC-ROC: {auc:.3f}")
print(f"Precision para atraso: {precision:.3f}")
print(f"Recall para atraso: {recall:.3f}")
print(f"F1-score para atraso: {f1:.3f}")
print("\nBASELINE: prever sempre 'no prazo'")
print(f"Precision baseline: {baseline_precision:.3f}")
print(f"Recall baseline: {baseline_recall:.3f}")

# Matriz de confusão
matriz = confusion_matrix(y_test, previsao)

print("\nMATRIZ DE CONFUSAO")
print(f"Verdadeiros negativos (no prazo correto): {matriz[0, 0]:,}")
print(f"Falsos positivos (alerta desnecessário): {matriz[0, 1]:,}")
print(f"Falsos negativos (atrasos não detectados): {matriz[1, 0]:,}")
print(f"Verdadeiros positivos (atrasos detectados): {matriz[1, 1]:,}")

# Salvar métricas
metricas = pd.DataFrame(
    {
        "metrica": [
            "AUC_ROC",
            "precision_atraso",
            "recall_atraso",
            "f1_atraso",
            "precision_baseline",
            "recall_baseline",
            "threshold",
            "pedidos_treino",
            "pedidos_teste",
            "taxa_atraso_teste",
        ],
        "valor": [
            auc,
            precision,
            recall,
            f1,
            baseline_precision,
            baseline_recall,
            threshold,
            len(treino),
            len(teste),
            y_test.mean(),
        ],
    }
)

metricas.to_csv(
    OUTPUT_DIR / "metricas_modelo_atraso.csv",
    index=False,
    encoding="utf-8-sig",
)

# Salvar previsões para score de risco e dashboard
resultado_teste = teste[
    [
        "order_id",
        "order_purchase_timestamp",
        "customer_state",
        "seller_state",
        "product_category_name",
        "preco_total",
        "frete_total",
        "itens",
        "product_weight_g",
        "delayed",
    ]
].copy()

resultado_teste["probabilidade_atraso"] = prob_atraso
resultado_teste["previsao_atraso"] = previsao

resultado_teste.to_csv(
    OUTPUT_DIR / "previsoes_atraso_teste.csv",
    index=False,
    encoding="utf-8-sig",
)

# Salvar matriz de confusão em CSV
pd.DataFrame(
    matriz,
    index=["Real: no prazo", "Real: atrasado"],
    columns=["Previsto: no prazo", "Previsto: atrasado"],
).to_csv(
    OUTPUT_DIR / "matriz_confusao.csv",
    encoding="utf-8-sig",
)

# Gerar gráfico da matriz
plt.figure(figsize=(6, 5))
sns.heatmap(
    matriz,
    annot=True,
    fmt=",",
    cmap="Blues",
    xticklabels=["Previsto: no prazo", "Previsto: atrasado"],
    yticklabels=["Real: no prazo", "Real: atrasado"],
)
plt.title("Modelo de atraso — Matriz de confusão")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "matriz_confusao.png",
    dpi=160,
)
plt.close()

# Curva de distribuição das probabilidades
plt.figure(figsize=(8, 5))
sns.histplot(
    data=pd.DataFrame(
        {
            "probabilidade_atraso": prob_atraso,
            "atrasou": y_test.map({0: "No prazo", 1: "Atrasou"}),
        }
    ),
    x="probabilidade_atraso",
    hue="atrasou",
    bins=30,
    stat="density",
    common_norm=False,
    element="step",
)
plt.title("Distribuição do score de atraso no período de teste")
plt.xlabel("Probabilidade prevista de atraso")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "distribuicao_scores_atraso.png",
    dpi=160,
)
plt.close()

print("\nArquivos gerados em: outputs/modelo_atraso/")
print("- metricas_modelo_atraso.csv")
print("- previsoes_atraso_teste.csv")
print("- matriz_confusao.csv")
print("- matriz_confusao.png")
print("- distribuicao_scores_atraso.png")
print("\nMODELO DE ATRASO CONCLUIDO.")