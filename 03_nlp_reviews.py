from pathlib import Path
import re
import unicodedata
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dados"
OUTPUT_DIR = BASE_DIR / "outputs" / "nlp_reviews"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def limpar_texto(texto):
    """Deixa o texto pronto para procurar palavras-chave."""

    if pd.isna(texto):
        return ""

    texto = str(texto).lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(
        caractere
        for caractere in texto
        if unicodedata.category(caractere) != "Mn"
    )
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


def contar_ocorrencias(texto, palavras):
    """Conta quantas palavras-chave de um tema aparecem no texto."""

    return sum(1 for palavra in palavras if palavra in texto)


# Dicionário simples, transparente e fácil de explicar no relatório.
# Uma review pode citar mais de um problema; aqui escolhemos o tema
# com maior número de ocorrências. Empate segue a prioridade operacional.
TEMAS = {
    "prazo_entrega": [
        "atraso",
        "atrasada",
        "atrasado",
        "demora",
        "demorou",
        "demorado",
        "prazo",
        "entrega",
        "nao chegou",
        "nao recebi",
        "ainda nao recebi",
        "ate hoje nao",
        "transportadora",
        "correios",
        "rastreio",
        "rastreamento",
        "chegar",
    ],
    "produto": [
        "produto",
        "defeito",
        "defeituoso",
        "nao funciona",
        "nao funcionou",
        "quebrado",
        "quebrada",
        "errado",
        "diferente",
        "qualidade",
        "falsificado",
        "falsa",
        "tamanho",
        "cor",
        "incompleto",
        "faltando",
        "usado",
    ],
    "atendimento": [
        "atendimento",
        "atendente",
        "suporte",
        "sac",
        "resposta",
        "respondem",
        "responde",
        "contato",
        "telefone",
        "email",
        "reclamacao",
        "ninguem",
        "descasso",
        "descaso",
        "vendedor nao",
    ],
    "embalagem_avaria": [
        "embalagem",
        "amassado",
        "amassada",
        "avariado",
        "avariada",
        "avaria",
        "danificado",
        "danificada",
        "rasgado",
        "rasgada",
        "quebrou no transporte",
        "sem protecao",
        "protecao",
        "caixa",
        "lacre",
    ],
}


def classificar_tema(texto):
    """Retorna tema, confiança simples e quantidade de palavras encontradas."""

    pontuacoes = {
        tema: contar_ocorrencias(texto, palavras)
        for tema, palavras in TEMAS.items()
    }

    maior_pontuacao = max(pontuacoes.values())

    if maior_pontuacao == 0:
        return "outros_sem_classificacao", "baixa", 0

    # Ordem usada quando duas categorias empatam
    prioridade = [
        "prazo_entrega",
        "produto",
        "embalagem_avaria",
        "atendimento",
    ]

    candidatos = [
        tema
        for tema in prioridade
        if pontuacoes[tema] == maior_pontuacao
    ]

    tema_escolhido = candidatos[0]

    # Confiança não é probabilidade estatística:
    # é apenas uma regra operacional para saber o que revisar.
    if maior_pontuacao >= 2:
        confianca = "alta"
    else:
        confianca = "media"

    return tema_escolhido, confianca, maior_pontuacao


print("=" * 70)
print("OLIST | 03 - TRIAGEM AUTOMATICA DE REVIEWS NEGATIVOS")
print("=" * 70)

arquivo_reviews = DATA_DIR / "olist_order_reviews_dataset.csv"

if not arquivo_reviews.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado: {arquivo_reviews}"
    )

reviews = pd.read_csv(arquivo_reviews)

# Apenas notas 1 e 2 são consideradas reviews negativas
negativos = reviews[reviews["review_score"] <= 2].copy()

# Juntar título e comentário; alguns registros não têm texto
negativos["texto_original"] = (
    negativos["review_comment_title"].fillna("")
    + " "
    + negativos["review_comment_message"].fillna("")
).str.strip()

negativos["texto_limpo"] = negativos["texto_original"].apply(limpar_texto)

resultado_classificacao = negativos["texto_limpo"].apply(classificar_tema)

negativos["tema"] = resultado_classificacao.apply(lambda x: x[0])
negativos["confianca_regra"] = resultado_classificacao.apply(lambda x: x[1])
negativos["palavras_chave_encontradas"] = resultado_classificacao.apply(
    lambda x: x[2]
)

# Indicador de review negativa com comentário aproveitável
negativos["tem_texto"] = negativos["texto_limpo"].str.len() > 0

# Resumo por tema
resumo_tema = (
    negativos.groupby("tema", as_index=False)
    .agg(
        reviews=("review_id", "count"),
        com_texto=("tem_texto", "sum"),
        confianca_alta=(
            "confianca_regra",
            lambda serie: (serie == "alta").sum()
        ),
    )
    .sort_values("reviews", ascending=False)
)

resumo_tema["percentual"] = (
    resumo_tema["reviews"] / len(negativos) * 100
)

# Resumo por nível de confiança
resumo_confianca = (
    negativos.groupby("confianca_regra", as_index=False)
    .agg(reviews=("review_id", "count"))
    .sort_values("reviews", ascending=False)
)

# Amostra para validação manual: 20 reviews por tema
amostra_validacao = (
    negativos[negativos["texto_limpo"].str.len() > 0]
    .groupby("tema", group_keys=False)
    .sample(n=20, replace=True, random_state=42)
    .sort_values("tema")
)

# Salvar dados
negativos.to_csv(
    OUTPUT_DIR / "reviews_negativos_classificados.csv",
    index=False,
    encoding="utf-8-sig",
)

resumo_tema.to_csv(
    OUTPUT_DIR / "resumo_por_tema.csv",
    index=False,
    encoding="utf-8-sig",
)

resumo_confianca.to_csv(
    OUTPUT_DIR / "resumo_confianca.csv",
    index=False,
    encoding="utf-8-sig",
)

amostra_validacao[
    [
        "review_id",
        "order_id",
        "review_score",
        "tema",
        "confianca_regra",
        "palavras_chave_encontradas",
        "texto_original",
    ]
].to_csv(
    OUTPUT_DIR / "amostra_para_validacao_manual.csv",
    index=False,
    encoding="utf-8-sig",
)

# Mostrar resultados principais
print(f"\nTotal de reviews negativos (nota 1 ou 2): {len(negativos):,}")
print(
    "Reviews negativos com texto: "
    f"{negativos['tem_texto'].sum():,} "
    f"({negativos['tem_texto'].mean():.1%})"
)

print("\nDISTRIBUICAO POR TEMA")
print("-" * 70)

for _, linha in resumo_tema.iterrows():
    print(
        f"{linha['tema']}: "
        f"{int(linha['reviews']):,} "
        f"({linha['percentual']:.1f}%)"
    )

print("\nDISTRIBUICAO POR CONFIANCA DA REGRA")
print("-" * 70)

for _, linha in resumo_confianca.iterrows():
    print(
        f"{linha['confianca_regra']}: "
        f"{int(linha['reviews']):,}"
    )

# Gráfico 1: reviews por tema
plot_temas = resumo_tema.sort_values("reviews", ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(plot_temas["tema"], plot_temas["reviews"])
plt.xlabel("Quantidade de reviews negativos")
plt.ylabel("Tema")
plt.title("Triagem automática — Reviews negativos por tema")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "reviews_por_tema.png",
    dpi=160,
)
plt.close()

# Gráfico 2: confiança
ordem_confianca = ["alta", "media", "baixa"]

plot_confianca = (
    resumo_confianca.set_index("confianca_regra")
    .reindex(ordem_confianca)
    .fillna(0)
    .reset_index()
)

plt.figure(figsize=(7, 5))
plt.bar(
    plot_confianca["confianca_regra"],
    plot_confianca["reviews"],
)
plt.xlabel("Confiança da regra")
plt.ylabel("Quantidade de reviews")
plt.title("Triagem automática — Confiança operacional")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "confianca_classificacao.png",
    dpi=160,
)
plt.close()

print("\nArquivos gerados em outputs/nlp_reviews/")
print("- reviews_negativos_classificados.csv")
print("- resumo_por_tema.csv")
print("- resumo_confianca.csv")
print("- amostra_para_validacao_manual.csv")
print("- reviews_por_tema.png")
print("- confianca_classificacao.png")

print("\nTRIAGEM DE REVIEWS CONCLUIDA.")