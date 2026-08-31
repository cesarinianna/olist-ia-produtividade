from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Olist | IA e Produtividade",
    page_icon="📦",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS = BASE_DIR / "outputs"


@st.cache_data
def carregar_csv(caminho):
    return pd.read_csv(caminho)


def mostrar_imagem(caminho, legenda):
    if caminho.exists():
        st.image(str(caminho), caption=legenda, width="stretch")
    else:
        st.warning(f"Gráfico não encontrado: {caminho.name}")


def ler_csv(caminho):
    if caminho.exists():
        return carregar_csv(caminho)
    return pd.DataFrame()


etapas = ler_csv(OUTPUTS / "diagnostico_etapas.csv")
atraso_mensal = ler_csv(OUTPUTS / "atraso_mensal.csv")
atraso_estado = ler_csv(OUTPUTS / "atraso_por_estado.csv")

metricas_atraso = ler_csv(
    OUTPUTS / "modelo_atraso" / "metricas_modelo_atraso.csv"
)

reviews_tema = ler_csv(
    OUTPUTS / "nlp_reviews" / "resumo_por_tema.csv"
)

reviews_confianca = ler_csv(
    OUTPUTS / "nlp_reviews" / "resumo_confianca.csv"
)

faixas_risco = ler_csv(
    OUTPUTS / "score_risco" / "resumo_faixas_risco.csv"
)

simulacao_risco = ler_csv(
    OUTPUTS / "score_risco" / "simulacao_por_faixa.csv"
)

cenarios = ler_csv(
    OUTPUTS / "simulacao_humano_maquina" / "cenarios_triagem_reviews.csv"
)

st.title("📦 Olist | Otimização da Malha Logística com IA")
st.caption(
    "Dashboard executivo — produtividade, logística, reviews e "
    "equilíbrio humano-máquina."
)

pagina = st.sidebar.radio(
    "Navegação",
    [
        "Resumo executivo",
        "Diagnóstico operacional",
        "Modelo de atraso",
        "Triagem de reviews",
        "Score de risco",
        "Humano x máquina",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("### Regra de decisão")
st.sidebar.write(
    "Automatizar tarefas repetitivas, auditáveis e reversíveis. "
    "Manter pessoas em casos ambíguos, reputacionais ou de alto impacto."
)

if pagina == "Resumo executivo":
    st.header("Decisão para o COO")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pedidos analisados", "99.441")
    col2.metric("Tempo médio compra-entrega", "12,6 dias")
    col3.metric("Reviews negativas", "14.575")
    col4.metric("Reviews negativas", "14,69%")

    st.success(
        "Recomendação: automatizar a triagem de reviews de alta confiança "
        "e o roteamento inicial de prazo/entrega e produto."
    )

    st.error(
        "Não usar ainda o modelo de atraso para enviar pedidos a uma fila "
        "humana: a precision foi de 4,6%, gerando muitos alertas falsos."
    )

    prioridades = pd.DataFrame(
        {
            "Iniciativa": [
                "Triagem de reviews de alta confiança",
                "Roteamento de prazo/entrega e produto",
                "Alertas de postagem e SLA ao vendedor",
                "Enriquecimento de dados logísticos",
                "Intervenção humana automática pelo score",
            ],
            "Impacto": [
                "Alto",
                "Alto",
                "Médio-alto",
                "Alto potencial",
                "Incerto",
            ],
            "Esforço": ["Baixo", "Médio", "Baixo", "Médio-alto", "Médio"],
            "Decisão": [
                "Prioridade 1",
                "Prioridade 2",
                "Prioridade 3",
                "Construir antes do novo modelo",
                "Não priorizar agora",
            ],
        }
    )

    st.dataframe(prioridades, width="stretch", hide_index=True)

    col1, col2 = st.columns(2)

    with col1:
        mostrar_imagem(
            OUTPUTS / "nlp_reviews" / "reviews_por_tema.png",
            "Reviews negativas por tema",
        )

    with col2:
        mostrar_imagem(
            OUTPUTS / "simulacao_humano_maquina" / "custo_cenarios_reviews.png",
            "Custo total estimado por cenário",
        )

elif pagina == "Diagnóstico operacional":
    st.header("Diagnóstico do ciclo do pedido")

    st.write(
        "Oportunidades de automação foram avaliadas por volume, "
        "repetitividade, clareza de regras, reversibilidade e impacto ao cliente."
    )

    if not etapas.empty:
        tabela = etapas.copy()

        for coluna in ["media_horas", "mediana_horas", "desvio_horas"]:
            if coluna in tabela.columns:
                tabela[coluna] = tabela[coluna].round(1)

        st.dataframe(tabela, width="stretch", hide_index=True)

    col1, col2 = st.columns(2)

    with col1:
        mostrar_imagem(
            OUTPUTS / "graficos" / "04_etapas.png",
            "Tempo médio por etapa operacional",
        )

    with col2:
        mostrar_imagem(
            OUTPUTS / "graficos" / "01_atraso_mensal.png",
            "Evolução mensal da taxa de atraso",
        )

    st.subheader("Atraso por estado do cliente")

    if not atraso_estado.empty:
        estados = atraso_estado.copy()
        estados["taxa_atraso_pct"] = estados["taxa_atraso"] * 100

        minimo = st.slider(
            "Mínimo de pedidos para exibir o estado",
            min_value=100,
            max_value=3000,
            value=100,
            step=100,
        )

        estados = estados[estados["pedidos"] >= minimo]

        st.bar_chart(
            estados.set_index("customer_state")["taxa_atraso_pct"]
        )

        st.dataframe(
            estados[
                ["customer_state", "pedidos", "taxa_atraso_pct"]
            ].sort_values("taxa_atraso_pct", ascending=False),
            width="stretch",
            hide_index=True,
        )

elif pagina == "Modelo de atraso":
    st.header("Modelo de previsão de atraso")

    st.warning(
        "O modelo foi validado temporalmente e não foi aprovado para "
        "intervenção humana automática no estágio atual."
    )

    metricas = {}

    if not metricas_atraso.empty:
        metricas = dict(
            zip(metricas_atraso["metrica"], metricas_atraso["valor"])
        )

    col1, col2, col3 = st.columns(3)
    col1.metric("AUC-ROC", f"{float(metricas.get('AUC_ROC', 0)):.3f}")
    col2.metric(
        "Precision — atraso",
        f"{float(metricas.get('precision_atraso', 0)):.1%}",
    )
    col3.metric(
        "Recall — atraso",
        f"{float(metricas.get('recall_atraso', 0)):.1%}",
    )

    st.write(
        "No teste temporal, o modelo gerou 3.669 falsos positivos e "
        "deixou passar 844 atrasos. A decisão é usar o resultado apenas "
        "para monitoramento até incluir dados de transportadora, rastreio, "
        "CEP/distância, estoque e SLA."
    )

    col1, col2 = st.columns(2)

    with col1:
        mostrar_imagem(
            OUTPUTS / "modelo_atraso" / "matriz_confusao.png",
            "Matriz de confusão no período de teste",
        )

    with col2:
        mostrar_imagem(
            OUTPUTS / "modelo_atraso" / "distribuicao_scores_atraso.png",
            "Distribuição das probabilidades previstas",
        )

elif pagina == "Triagem de reviews":
    st.header("Triagem automática de reviews negativos")

    st.write(
        "A abordagem usa regras e palavras-chave em português. "
        "É barata, auditável e adequada para um piloto, sem necessidade "
        "de API paga, GPU ou uma base rotulada."
    )

    if not reviews_tema.empty:
        total_reviews = int(reviews_tema["reviews"].sum())

        col1, col2, col3 = st.columns(3)
        col1.metric("Reviews negativas", f"{total_reviews:,}".replace(",", "."))
        col2.metric("Prazo + produto", "58,5%")
        col3.metric("Casos ambíguos", "37,3%")

        st.dataframe(
            reviews_tema.sort_values("reviews", ascending=False),
            width="stretch",
            hide_index=True,
        )

    col1, col2 = st.columns(2)

    with col1:
        mostrar_imagem(
            OUTPUTS / "nlp_reviews" / "reviews_por_tema.png",
            "Distribuição dos temas",
        )

    with col2:
        mostrar_imagem(
            OUTPUTS / "nlp_reviews" / "confianca_classificacao.png",
            "Confiança operacional da triagem",
        )

    if not reviews_confianca.empty:
        st.subheader("Fila recomendada")

        filas = pd.DataFrame(
            {
                "Nível": [
                    "Alta confiança",
                    "Média confiança",
                    "Baixa confiança",
                ],
                "Tratamento": [
                    "Automação no piloto",
                    "Auditoria/amostra antes de automatizar",
                    "Revisão humana obrigatória",
                ],
            }
        )

        st.dataframe(filas, width="stretch", hide_index=True)

elif pagina == "Score de risco":
    st.header("Score de risco no checkout")

    st.info(
        "O score combina risco histórico de rota, categoria, origem/destino, "
        "peso, frete e quantidade de itens. É explicável, mas ainda não "
        "diferencia risco o suficiente para uma fila humana automática."
    )

    if not faixas_risco.empty:
        st.dataframe(faixas_risco, width="stretch", hide_index=True)

    if not simulacao_risco.empty:
        st.subheader("Validação por faixa")

        st.dataframe(
            simulacao_risco,
            width="stretch",
            hide_index=True,
        )

    col1, col2 = st.columns(2)

    with col1:
        mostrar_imagem(
            OUTPUTS / "score_risco" / "distribuicao_score_risco.png",
            "Distribuição do score",
        )

    with col2:
        mostrar_imagem(
            OUTPUTS / "score_risco" / "pedidos_por_faixa_risco.png",
            "Pedidos por faixa de risco",
        )

elif pagina == "Humano x máquina":
    st.header("Simulação: humano x máquina")

    st.write(
        "A simulação usa hipóteses explícitas de tempo por review, "
        "custo/hora de atendimento e custo de erro. Em produção, esses "
        "parâmetros devem ser substituídos por dados financeiros da Olist."
    )

    if not cenarios.empty:
        melhor = cenarios.loc[cenarios["custo_total_estimado"].idxmin()]

        col1, col2, col3 = st.columns(3)
        col1.metric("Melhor cenário", "Híbrido: só alta confiança")
        col2.metric(
            "Custo estimado",
            f"R$ {melhor['custo_total_estimado']:,.2f}".replace(
                ",", "X"
            ).replace(".", ",").replace("X", "."),
        )
        col3.metric(
            "Horas poupadas",
            f"{melhor['horas_economizadas_vs_manual']:.1f}".replace(".", ","),
        )

        st.dataframe(cenarios, width="stretch", hide_index=True)

    col1, col2 = st.columns(2)

    with col1:
        mostrar_imagem(
            OUTPUTS / "simulacao_humano_maquina" / "custo_cenarios_reviews.png",
            "Custo total estimado por cenário",
        )

    with col2:
        mostrar_imagem(
            OUTPUTS / "simulacao_humano_maquina" / "horas_cenarios_reviews.png",
            "Horas operacionais por cenário",
        )

    st.success(
        "Decisão: automatizar apenas reviews de alta confiança no piloto. "
        "Manter humano em baixa/média confiança, atendimento, avaria e "
        "casos ambíguos."
    )

st.divider()
st.caption(
    "Fonte: Brazilian E-Commerce Public Dataset by Olist. "
    "Projeto acadêmico; custos são hipóteses documentadas."
)