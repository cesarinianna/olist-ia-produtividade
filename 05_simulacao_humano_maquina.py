from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent

NLP_FILE = (
    BASE_DIR
    / "outputs"
    / "nlp_reviews"
    / "reviews_negativos_classificados.csv"
)

ATRASO_FILE = (
    BASE_DIR
    / "outputs"
    / "modelo_atraso"
    / "previsoes_atraso_teste.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "simulacao_humano_maquina"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def custo_horas(
    quantidade_automatica,
    quantidade_humana,
    minutos_automatico,
    minutos_humano,
):
    """
    Calcula esforço mensal em horas.
    A ação automática representa checagem/roteamento.
    A ação humana representa leitura + decisão + encaminhamento.
    """

    minutos_total = (
        quantidade_automatica * minutos_automatico
        + quantidade_humana * minutos_humano
    )

    return minutos_total / 60


print("=" * 70)
print("OLIST | 05 - SIMULACAO HUMANO X MAQUINA")
print("=" * 70)

if not NLP_FILE.exists():
    raise FileNotFoundError(
        "Arquivo de NLP não encontrado. Rode 03_nlp_reviews.py."
    )

if not ATRASO_FILE.exists():
    raise FileNotFoundError(
        "Arquivo de atraso não encontrado. Rode 02_modelo_atraso.py."
    )

reviews = pd.read_csv(NLP_FILE)
previsoes_atraso = pd.read_csv(ATRASO_FILE)

# ------------------------------------------------------------
# PARÂMETROS DE NEGÓCIO — hipóteses explícitas.
# Você pode explicar no relatório que estes custos devem ser
# substituídos por dados reais da Olist em uma implantação.
# ------------------------------------------------------------

MINUTOS_TRIAGEM_MANUAL = 4
MINUTOS_TRIAGEM_AUTOMATICA = 0.25

CUSTO_HORA_CX = 35.00

# Custo médio de uma review negativa que não é priorizada.
# É uma hipótese de proxy: retrabalho, contato, cupom,
# risco de recompra menor e impacto em reputação.
CUSTO_FALSO_NEGATIVO_REVIEW = 25.00

# Custo de abrir uma fila humana de forma desnecessária
CUSTO_FALSO_POSITIVO_ALERTA = 8.00

# Custo operacional estimado de atraso não identificado
CUSTO_ATRASO_NAO_DETECTADO = 35.00

# ------------------------------------------------------------
# CENÁRIOS DE TRIAGEM DE REVIEWS
# ------------------------------------------------------------

total_reviews = len(reviews)

# Confiança alta: pode ser automatizada.
alta = int((reviews["confianca_regra"] == "alta").sum())

# Média: pode ser encaminhada automaticamente, mas com auditoria.
media = int((reviews["confianca_regra"] == "media").sum())

# Baixa: vai para humano; inclui "outros" e reviews sem texto.
baixa = int((reviews["confianca_regra"] == "baixa").sum())

# Cenário 1: tudo manual
horas_tudo_manual = custo_horas(
    quantidade_automatica=0,
    quantidade_humana=total_reviews,
    minutos_automatico=MINUTOS_TRIAGEM_AUTOMATICA,
    minutos_humano=MINUTOS_TRIAGEM_MANUAL,
)

# Cenário 2: 100% automático.
# Os casos com baixa confiança representam risco de mau roteamento.
horas_100_auto = custo_horas(
    quantidade_automatica=total_reviews,
    quantidade_humana=0,
    minutos_automatico=MINUTOS_TRIAGEM_AUTOMATICA,
    minutos_humano=MINUTOS_TRIAGEM_MANUAL,
)

custo_erro_100_auto = baixa * CUSTO_FALSO_NEGATIVO_REVIEW

# Cenário 3: Automação somente de alta confiança.
horas_hibrido_alta = custo_horas(
    quantidade_automatica=alta,
    quantidade_humana=total_reviews - alta,
    minutos_automatico=MINUTOS_TRIAGEM_AUTOMATICA,
    minutos_humano=MINUTOS_TRIAGEM_MANUAL,
)

# Suposição: mesmo regras de alta confiança podem errar 5%.
falsos_negativos_hibrido_alta = round(alta * 0.05)

custo_erro_hibrido_alta = (
    falsos_negativos_hibrido_alta
    * CUSTO_FALSO_NEGATIVO_REVIEW
)

# Cenário 4: Automação alta + média, humano em baixa confiança.
horas_hibrido_alta_media = custo_horas(
    quantidade_automatica=alta + media,
    quantidade_humana=baixa,
    minutos_automatico=MINUTOS_TRIAGEM_AUTOMATICA,
    minutos_humano=MINUTOS_TRIAGEM_MANUAL,
)

# Hipótese: erros de alta confiança 5% e de média 15%.
falsos_negativos_hibrido_alta_media = round(
    alta * 0.05 + media * 0.15
)

custo_erro_hibrido_alta_media = (
    falsos_negativos_hibrido_alta_media
    * CUSTO_FALSO_NEGATIVO_REVIEW
)

cenarios_reviews = pd.DataFrame(
    {
        "cenario": [
            "100% manual",
            "100% automatico",
            "Hibrido: automatizar apenas alta confianca",
            "Hibrido: automatizar alta + media; humano na baixa",
        ],
        "reviews_automaticas": [
            0,
            total_reviews,
            alta,
            alta + media,
        ],
        "reviews_humanas": [
            total_reviews,
            0,
            total_reviews - alta,
            baixa,
        ],
        "horas_operacionais": [
            horas_tudo_manual,
            horas_100_auto,
            horas_hibrido_alta,
            horas_hibrido_alta_media,
        ],
        "falsos_negativos_estimados": [
            0,
            baixa,
            falsos_negativos_hibrido_alta,
            falsos_negativos_hibrido_alta_media,
        ],
        "custo_erros_estimado": [
            0,
            custo_erro_100_auto,
            custo_erro_hibrido_alta,
            custo_erro_hibrido_alta_media,
        ],
    }
)

cenarios_reviews["horas_economizadas_vs_manual"] = (
    horas_tudo_manual
    - cenarios_reviews["horas_operacionais"]
)

cenarios_reviews["custo_mao_de_obra"] = (
    cenarios_reviews["horas_operacionais"]
    * CUSTO_HORA_CX
)

cenarios_reviews["custo_total_estimado"] = (
    cenarios_reviews["custo_mao_de_obra"]
    + cenarios_reviews["custo_erros_estimado"]
)

# ------------------------------------------------------------
# CENÁRIOS PARA MODELO DE ATRASO
# ------------------------------------------------------------

# O resultado é o conjunto de teste temporal criado no script 02.
# 1 = atraso, 0 = no prazo.
y_real = previsoes_atraso["delayed"]
y_pred = previsoes_atraso["previsao_atraso"]

falsos_positivos = int(((y_pred == 1) & (y_real == 0)).sum())
falsos_negativos = int(((y_pred == 0) & (y_real == 1)).sum())
verdadeiros_positivos = int(((y_pred == 1) & (y_real == 1)).sum())

# Horas por alerta humano: investigar pedido, consultar vendedor
# e registrar ação. Hipótese para simulação.
MINUTOS_POR_ALERTA_LOGISTICO = 8

horas_alertas_modelo = (
    (falsos_positivos + verdadeiros_positivos)
    * MINUTOS_POR_ALERTA_LOGISTICO
    / 60
)

custo_alertas_modelo = horas_alertas_modelo * CUSTO_HORA_CX

custo_atrasos_nao_detectados = (
    falsos_negativos
    * CUSTO_ATRASO_NAO_DETECTADO
)

simulacao_atrasos = pd.DataFrame(
    {
        "metrica": [
            "Alertas totais para humano",
            "Verdadeiros atrasos detectados",
            "Falsos positivos",
            "Falsos negativos",
            "Horas humanas exigidas",
            "Custo de triagem humana",
            "Custo estimado dos atrasos nao detectados",
            "Custo total estimado",
        ],
        "valor": [
            falsos_positivos + verdadeiros_positivos,
            verdadeiros_positivos,
            falsos_positivos,
            falsos_negativos,
            horas_alertas_modelo,
            custo_alertas_modelo,
            custo_atrasos_nao_detectados,
            custo_alertas_modelo + custo_atrasos_nao_detectados,
        ],
    }
)

# ------------------------------------------------------------
# SALVAR RESULTADOS
# ------------------------------------------------------------

cenarios_reviews.to_csv(
    OUTPUT_DIR / "cenarios_triagem_reviews.csv",
    index=False,
    encoding="utf-8-sig",
)

simulacao_atrasos.to_csv(
    OUTPUT_DIR / "simulacao_modelo_atraso.csv",
    index=False,
    encoding="utf-8-sig",
)

# Gráfico: custo total por cenário de review
plot_reviews = cenarios_reviews.sort_values(
    "custo_total_estimado",
    ascending=True,
)

plt.figure(figsize=(11, 6))

plt.barh(
    plot_reviews["cenario"],
    plot_reviews["custo_total_estimado"],
)

plt.xlabel("Custo total estimado (R$)")
plt.ylabel("Cenário")
plt.title("Humano x máquina — Custo estimado da triagem de reviews")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "custo_cenarios_reviews.png",
    dpi=160,
)

plt.close()

# Gráfico: horas por cenário de review
plt.figure(figsize=(11, 6))

plt.barh(
    plot_reviews["cenario"],
    plot_reviews["horas_operacionais"],
)

plt.xlabel("Horas operacionais estimadas")
plt.ylabel("Cenário")
plt.title("Humano x máquina — Horas necessárias para triagem")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "horas_cenarios_reviews.png",
    dpi=160,
)

plt.close()

# ------------------------------------------------------------
# MOSTRAR RESULTADOS
# ------------------------------------------------------------

print("\nCENARIOS DE TRIAGEM DE REVIEWS")
print("-" * 70)

for _, linha in cenarios_reviews.iterrows():
    print(f"\n{linha['cenario']}")
    print(
        f"  Reviews automaticas: "
        f"{int(linha['reviews_automaticas']):,}"
    )
    print(
        f"  Reviews humanas: "
        f"{int(linha['reviews_humanas']):,}"
    )
    print(
        f"  Horas operacionais: "
        f"{linha['horas_operacionais']:.1f}"
    )
    print(
        f"  Horas economizadas vs manual: "
        f"{linha['horas_economizadas_vs_manual']:.1f}"
    )
    print(
        f"  Falsos negativos estimados: "
        f"{int(linha['falsos_negativos_estimados']):,}"
    )
    print(
        f"  Custo total estimado: "
        f"R$ {linha['custo_total_estimado']:,.2f}"
    )

print("\n" + "-" * 70)
print("SIMULACAO DO MODELO DE ATRASO")
print("-" * 70)

for _, linha in simulacao_atrasos.iterrows():
    if "Custo" in linha["metrica"]:
        print(f"{linha['metrica']}: R$ {linha['valor']:,.2f}")
    else:
        print(f"{linha['metrica']}: {linha['valor']:,.1f}")

print("\nRECOMENDACAO:")
print(
    "Automatizar reviews de alta e media confianca; "
    "manter humano nos casos de baixa confianca. "
    "Nao usar o modelo de atraso para acionar fila humana "
    "até enriquecer os dados e melhorar a precision."
)

print("\nArquivos gerados em outputs/simulacao_humano_maquina/")
print("- cenarios_triagem_reviews.csv")
print("- simulacao_modelo_atraso.csv")
print("- custo_cenarios_reviews.png")
print("- horas_cenarios_reviews.png")
print("\nSIMULACAO CONCLUIDA.")