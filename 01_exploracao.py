from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dados"
OUTPUT_DIR = BASE_DIR / "outputs"
CHART_DIR = OUTPUT_DIR / "graficos"
OUTPUT_DIR.mkdir(exist_ok=True)
CHART_DIR.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")


def carregar(nome):
    caminho = DATA_DIR / nome
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    return pd.read_csv(caminho)


print("=" * 65)
print("OLIST - ANALISE EXPLORATORIA")
print("=" * 65)

orders = carregar("olist_orders_dataset.csv")
items = carregar("olist_order_items_dataset.csv")
reviews = carregar("olist_order_reviews_dataset.csv")
customers = carregar("olist_customers_dataset.csv")
sellers = carregar("olist_sellers_dataset.csv")
products = carregar("olist_products_dataset.csv")
translation = carregar("product_category_name_translation.csv")

print(f"Pedidos: {len(orders):,}")
print(f"Itens: {len(items):,}")
print(f"Reviews: {len(reviews):,}")
print(f"Clientes: {len(customers):,}")
print(f"Vendedores: {len(sellers):,}")
print(f"Produtos: {len(products):,}")

# Datas
colunas_data = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

for coluna in colunas_data:
    orders[coluna] = pd.to_datetime(orders[coluna], errors="coerce")

entregues = orders[orders["order_delivered_customer_date"].notna()].copy()
entregues["delayed"] = (
    entregues["order_delivered_customer_date"]
    > entregues["order_estimated_delivery_date"]
).astype(int)

orders["approval_hours"] = (
    orders["order_approved_at"] - orders["order_purchase_timestamp"]
).dt.total_seconds() / 3600
orders["posting_hours"] = (
    orders["order_delivered_carrier_date"] - orders["order_approved_at"]
).dt.total_seconds() / 3600
orders["delivery_hours"] = (
    orders["order_delivered_customer_date"]
    - orders["order_delivered_carrier_date"]
).dt.total_seconds() / 3600
orders["end_to_end_days"] = (
    orders["order_delivered_customer_date"]
    - orders["order_purchase_timestamp"]
).dt.total_seconds() / 86400
orders["purchase_month"] = orders["order_purchase_timestamp"].dt.to_period("M").astype(str)

# Resumo executivo
atrasados = int(entregues["delayed"].sum())
taxa_atraso = entregues["delayed"].mean()
negativos = int((reviews["review_score"] <= 2).sum())
taxa_negativos = (reviews["review_score"] <= 2).mean()

print("\nKPIs EXECUTIVOS")
print("-" * 40)
print(f"Total de pedidos: {len(orders):,}")
print(f"Pedidos entregues: {len(entregues):,}")
print(f"Pedidos atrasados: {atrasados:,}")
print(f"Taxa de atraso: {taxa_atraso:.2%}")
print(f"Total de reviews: {len(reviews):,}")
print(f"Reviews negativos: {negativos:,}")
print(f"Taxa de reviews negativos: {taxa_negativos:.2%}")
print(f"Tempo medio compra-entrega: {orders.loc[entregues.index, 'end_to_end_days'].mean():.1f} dias")

# Diagnostico das etapas
etapas = pd.DataFrame({
    "etapa": ["Compra-aprovacao", "Aprovacao-postagem", "Postagem-entrega", "Compra-entrega"],
    "volume": [
        orders["approval_hours"].notna().sum(),
        orders["posting_hours"].notna().sum(),
        orders["delivery_hours"].notna().sum(),
        orders["end_to_end_days"].notna().sum(),
    ],
    "media_horas": [
        orders["approval_hours"].mean(),
        orders["posting_hours"].mean(),
        orders["delivery_hours"].mean(),
        orders["end_to_end_days"].mean() * 24,
    ],
    "mediana_horas": [
        orders["approval_hours"].median(),
        orders["posting_hours"].median(),
        orders["delivery_hours"].median(),
        orders["end_to_end_days"].median() * 24,
    ],
    "desvio_horas": [
        orders["approval_hours"].std(),
        orders["posting_hours"].std(),
        orders["delivery_hours"].std(),
        orders["end_to_end_days"].std() * 24,
    ],
})
etapas.to_csv(OUTPUT_DIR / "diagnostico_etapas.csv", index=False, encoding="utf-8-sig")

# Dados por estado e por mes
base_estado = entregues.merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left")
atraso_estado = base_estado.groupby("customer_state", as_index=False).agg(
    pedidos=("order_id", "count"), taxa_atraso=("delayed", "mean")
).query("pedidos >= 100").sort_values("taxa_atraso", ascending=False)
atraso_estado.to_csv(OUTPUT_DIR / "atraso_por_estado.csv", index=False, encoding="utf-8-sig")

entregues["purchase_month"] = entregues["order_purchase_timestamp"].dt.to_period("M").astype(str)
mensal = entregues.groupby("purchase_month", as_index=False).agg(
    pedidos=("order_id", "count"), taxa_atraso=("delayed", "mean")
)
mensal.to_csv(OUTPUT_DIR / "atraso_mensal.csv", index=False, encoding="utf-8-sig")

# Base consolidada simples para os modelos
items_pedido = items.groupby("order_id", as_index=False).agg(
    preco_total=("price", "sum"), frete_total=("freight_value", "sum"),
    itens=("order_item_id", "count"), seller_id=("seller_id", "first"), product_id=("product_id", "first")
)
base = orders.merge(items_pedido, on="order_id", how="left")
base = base.merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left")
base = base.merge(sellers[["seller_id", "seller_state"]], on="seller_id", how="left")
base = base.merge(products[["product_id", "product_category_name", "product_weight_g"]], on="product_id", how="left")
base.to_csv(OUTPUT_DIR / "base_pedidos_modelagem.csv", index=False, encoding="utf-8-sig")

# Graficos
plt.figure(figsize=(11, 5))
plt.plot(mensal["purchase_month"], mensal["taxa_atraso"] * 100, marker="o")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Taxa de atraso (%)")
plt.title("Taxa mensal de atraso")
plt.tight_layout()
plt.savefig(CHART_DIR / "01_atraso_mensal.png", dpi=150)
plt.close()

plot_estado = atraso_estado.head(10).sort_values("taxa_atraso")
plt.figure(figsize=(9, 5))
plt.barh(plot_estado["customer_state"], plot_estado["taxa_atraso"] * 100)
plt.xlabel("Taxa de atraso (%)")
plt.title("Estados com maior taxa de atraso")
plt.tight_layout()
plt.savefig(CHART_DIR / "02_atraso_estado.png", dpi=150)
plt.close()

notas = reviews["review_score"].value_counts().sort_index()
plt.figure(figsize=(7, 4))
plt.bar(notas.index.astype(str), notas.values)
plt.xlabel("Nota")
plt.ylabel("Reviews")
plt.title("Distribuicao das notas")
plt.tight_layout()
plt.savefig(CHART_DIR / "03_reviews.png", dpi=150)
plt.close()

plt.figure(figsize=(9, 4))
plt.bar(etapas["etapa"], etapas["media_horas"] / 24)
plt.xticks(rotation=15, ha="right")
plt.ylabel("Dias")
plt.title("Tempo medio por etapa")
plt.tight_layout()
plt.savefig(CHART_DIR / "04_etapas.png", dpi=150)
plt.close()

print("\nConcluido. Arquivos salvos na pasta outputs.")
