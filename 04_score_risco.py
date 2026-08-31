from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "outputs" / "base_pedidos_modelagem.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "score_risco"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def normalizar_0_100(serie):
    """Converte uma série em uma escala de 0 a 100."""

    minimo = serie.min()
    maximo = serie.max()

    if pd.isna(minimo) or pd.isna(maximo) or minimo == maximo:
        return pd.Series(50, index=serie.index)

    return 100 * (serie - minimo) / (maximo - minimo)


def faixa_risco(score):
    """Define ação operacional conforme faixa de risco."""

    if score < 30:
        return "Baixo - fluxo automatizado"
    elif score < 60:
        return "Medio - monitoramento automatico"
    elif score < 80:
        return "Alto - alerta operacional"
    else:
        return "Critico - revisao humana prioritaria"


print("=" * 70)
print("OLIST | 04 - SCORE DE RISCO POR PEDIDO")
print("=" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "Base não encontrada. Rode primeiro 01_exploracao.py."
    )

df = pd.read_csv(INPUT_FILE, low_memory=False)

# Preparar as datas
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

# Para calcular risco histórico, usamos pedidos entregues que têm resultado conhecido
historico = df[
    df["order_delivered_customer_date"].notna()
].copy()

historico["delayed"] = (
    historico["order_delivered_customer_date"]
    > historico["order_estimated_delivery_date"]
).astype(int)

# A data de corte representa o fim do treinamento histórico.
# Pedidos posteriores serão apenas pontuados, sem usar seu resultado futuro.
df = df.sort_values("order_purchase_timestamp").copy()
data_corte = df["order_purchase_timestamp"].quantile(0.80)

base_historica = historico[
    historico["order_purchase_timestamp"] <= data_corte
].copy()

pedidos_score = df[
    df["order_purchase_timestamp"] > data_corte
].copy()

# Se não houver pedidos após corte por algum problema, usar toda a base para demonstração
if len(pedidos_score) == 0:
    pedidos_score = df.copy()

print(f"Pedidos usados como histórico: {len(base_historica):,}")
print(f"Pedidos pontuados: {len(pedidos_score):,}")
print(f"Data de corte: {data_corte.date()}")

# Criar chave de rota
base_historica["rota"] = (
    base_historica["seller_state"].fillna("unknown")
    + "_"
    + base_historica["customer_state"].fillna("unknown")
)

pedidos_score["rota"] = (
    pedidos_score["seller_state"].fillna("unknown")
    + "_"
    + pedidos_score["customer_state"].fillna("unknown")
)

taxa_geral_atraso = base_historica["delayed"].mean()

# Risco histórico por rota
risco_rota = (
    base_historica.groupby("rota", as_index=False)
    .agg(
        pedidos_historicos_rota=("order_id", "count"),
        taxa_atraso_rota=("delayed", "mean"),
    )
)

# Suavização: rotas com poucos pedidos não podem dominar o score
risco_rota["taxa_atraso_rota_suavizada"] = (
    risco_rota["taxa_atraso_rota"] * risco_rota["pedidos_historicos_rota"]
    + taxa_geral_atraso * 30
) / (risco_rota["pedidos_historicos_rota"] + 30)

# Risco histórico por categoria
risco_categoria = (
    base_historica.groupby(
        "product_category_name",
        as_index=False,
    )
    .agg(
        pedidos_historicos_categoria=("order_id", "count"),
        taxa_atraso_categoria=("delayed", "mean"),
    )
)

risco_categoria["taxa_atraso_categoria_suavizada"] = (
    risco_categoria["taxa_atraso_categoria"]
    * risco_categoria["pedidos_historicos_categoria"]
    + taxa_geral_atraso * 30
) / (risco_categoria["pedidos_historicos_categoria"] + 30)

# Adicionar riscos históricos à base que será pontuada
score = pedidos_score.merge(
    risco_rota[
        [
            "rota",
            "taxa_atraso_rota_suavizada",
            "pedidos_historicos_rota",
        ]
    ],
    on="rota",
    how="left",
)

score = score.merge(
    risco_categoria[
        [
            "product_category_name",
            "taxa_atraso_categoria_suavizada",
            "pedidos_historicos_categoria",
        ]
    ],
    on="product_category_name",
    how="left",
)

# Onde não há histórico suficiente, usar a taxa média da operação
score["taxa_atraso_rota_suavizada"] = (
    score["taxa_atraso_rota_suavizada"]
    .fillna(taxa_geral_atraso)
)

score["taxa_atraso_categoria_suavizada"] = (
    score["taxa_atraso_categoria_suavizada"]
    .fillna(taxa_geral_atraso)
)

# Sinais conhecidos no checkout
score["mesmo_estado"] = (
    score["customer_state"].fillna("unknown")
    == score["seller_state"].fillna("unknown")
).astype(int)

score["risco_fora_estado"] = (
    1 - score["mesmo_estado"]
) * 100

# Peso alto: usar percentil 75 da base histórica
limite_peso = base_historica["product_weight_g"].quantile(0.75)

score["risco_peso"] = (
    score["product_weight_g"]
    .fillna(base_historica["product_weight_g"].median())
    .clip(lower=0)
)

score["risco_peso"] = normalizar_0_100(score["risco_peso"])

# Frete alto: também é sinal de rota/complexidade
score["risco_frete"] = (
    score["frete_total"]
    .fillna(base_historica["frete_total"].median())
    .clip(lower=0)
)

score["risco_frete"] = normalizar_0_100(score["risco_frete"])

# Mais itens pode aumentar risco de separação e postagem
score["risco_itens"] = normalizar_0_100(
    score["itens"].fillna(1).clip(lower=1)
)

# Componentes finais. Pesos explícitos e explicáveis.
score["componente_rota"] = (
    score["taxa_atraso_rota_suavizada"] * 100 * 0.35
)

score["componente_categoria"] = (
    score["taxa_atraso_categoria_suavizada"] * 100 * 0.20
)

score["componente_fora_estado"] = (
    score["risco_fora_estado"] * 0.20
)

score["componente_peso"] = (
    score["risco_peso"] * 0.15
)

score["componente_frete"] = (
    score["risco_frete"] * 0.05
)

score["componente_itens"] = (
    score["risco_itens"] * 0.05
)

score["score_risco"] = (
    score["componente_rota"]
    + score["componente_categoria"]
    + score["componente_fora_estado"]
    + score["componente_peso"]
    + score["componente_frete"]
    + score["componente_itens"]
).clip(0, 100)

score["score_risco"] = score["score_risco"].round(1)

score["faixa_risco"] = score["score_risco"].apply(faixa_risco)

# Manter colunas úteis no arquivo final
colunas_saida = [
    "order_id",
    "order_purchase_timestamp",
    "customer_state",
    "seller_state",
    "rota",
    "product_category_name",
    "preco_total",
    "frete_total",
    "itens",
    "product_weight_g",
    "taxa_atraso_rota_suavizada",
    "taxa_atraso_categoria_suavizada",
    "score_risco",
    "faixa_risco",
    "componente_rota",
    "componente_categoria",
    "componente_fora_estado",
    "componente_peso",
    "componente_frete",
    "componente_itens",
]

score_final = score[
    [coluna for coluna in colunas_saida if coluna in score.columns]
].copy()

# Se o pedido já tiver sido entregue, manter a informação real para validação
if "order_delivered_customer_date" in score.columns:
    score_final["order_delivered_customer_date"] = (
        score["order_delivered_customer_date"]
    )
    score_final["order_estimated_delivery_date"] = (
        score["order_estimated_delivery_date"]
    )
    score_final["atrasou_real"] = (
        score["order_delivered_customer_date"]
        > score["order_estimated_delivery_date"]
    ).astype("Int64")

# Resumo de faixas
resumo_faixas = (
    score_final.groupby("faixa_risco", as_index=False)
    .agg(
        pedidos=("order_id", "count"),
        score_medio=("score_risco", "mean"),
    )
)

resumo_faixas["percentual"] = (
    resumo_faixas["pedidos"] / len(score_final) * 100
)

# Simulação operacional, se houver resultado real dos pedidos
if "atrasou_real" in score_final.columns:
    simulacao = (
        score_final.groupby("faixa_risco", as_index=False)
        .agg(
            pedidos=("order_id", "count"),
            atrasos_reais=("atrasou_real", "sum"),
            taxa_atraso_real=("atrasou_real", "mean"),
        )
    )
else:
    simulacao = resumo_faixas.copy()

# Salvar
score_final.to_csv(
    OUTPUT_DIR / "pedidos_com_score_risco.csv",
    index=False,
    encoding="utf-8-sig",
)

resumo_faixas.to_csv(
    OUTPUT_DIR / "resumo_faixas_risco.csv",
    index=False,
    encoding="utf-8-sig",
)

simulacao.to_csv(
    OUTPUT_DIR / "simulacao_por_faixa.csv",
    index=False,
    encoding="utf-8-sig",
)

# Mostrar resultados
print("\nDISTRIBUICAO DO SCORE")
print("-" * 70)

for _, linha in resumo_faixas.iterrows():
    print(
        f"{linha['faixa_risco']}: "
        f"{int(linha['pedidos']):,} pedidos | "
        f"score médio {linha['score_medio']:.1f} | "
        f"{linha['percentual']:.1f}%"
    )

if "atrasou_real" in score_final.columns:
    print("\nVALIDACAO OPERACIONAL POR FAIXA")
    print("-" * 70)

    for _, linha in simulacao.iterrows():
        print(
            f"{linha['faixa_risco']}: "
            f"{int(linha['atrasos_reais']):,} atrasos em "
            f"{int(linha['pedidos']):,} pedidos | "
            f"taxa real de atraso {linha['taxa_atraso_real']:.2%}"
        )

# Gráfico: distribuição de score
plt.figure(figsize=(10, 6))

sns.histplot(
    score_final["score_risco"],
    bins=30,
    kde=True,
)

plt.axvline(
    30,
    color="green",
    linestyle="--",
    label="Baixo/Médio",
)

plt.axvline(
    60,
    color="orange",
    linestyle="--",
    label="Médio/Alto",
)

plt.axvline(
    80,
    color="red",
    linestyle="--",
    label="Alto/Crítico",
)

plt.xlabel("Score de risco")
plt.ylabel("Quantidade de pedidos")
plt.title("Distribuição do score de risco por pedido")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "distribuicao_score_risco.png",
    dpi=160,
)

plt.close()

# Gráfico: pedidos por faixa
ordem_faixas = [
    "Baixo - fluxo automatizado",
    "Medio - monitoramento automatico",
    "Alto - alerta operacional",
    "Critico - revisao humana prioritaria",
]

plot_faixas = (
    resumo_faixas.set_index("faixa_risco")
    .reindex(ordem_faixas)
    .fillna(0)
    .reset_index()
)

plt.figure(figsize=(10, 6))

plt.barh(
    plot_faixas["faixa_risco"],
    plot_faixas["pedidos"],
)

plt.xlabel("Quantidade de pedidos")
plt.ylabel("Faixa de risco")
plt.title("Pedidos por faixa de risco")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "pedidos_por_faixa_risco.png",
    dpi=160,
)

plt.close()

print("\nArquivos gerados em outputs/score_risco/")
print("- pedidos_com_score_risco.csv")
print("- resumo_faixas_risco.csv")
print("- simulacao_por_faixa.csv")
print("- distribuicao_score_risco.png")
print("- pedidos_por_faixa_risco.png")

print("\nSCORE DE RISCO CONCLUIDO.")