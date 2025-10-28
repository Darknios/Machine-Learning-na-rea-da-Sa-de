import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# =============================
# CONFIGURAÇÕES INICIAIS
# =============================

st.set_page_config(page_title="Previsão de Doenças Cardíacas", layout="wide")
st.title("Projeto de Machine Learning Aplicado à Saúde Cardiovascular")

st.markdown("""
Aplicativo interativo desenvolvido para:

1. Visualizar a **análise descritiva** dos dados clínicos.
2. Realizar **previsões reais** com base no dataset *Heart Disease*.
3. Explorar **agrupamentos de pacientes** com base em similaridades clínicas.
   """)

# =============================
# LEITURA DO DATASET REAL
# =============================

caminho = "heart.csv"
df = pd.read_csv(caminho)

# =============================
# FILTROS PARA ANÁLISE EXPLORATÓRIA
# =============================

def criar_filtros_analise():
    st.sidebar.header("Filtros para Análise Exploratória")
    
    # Filtro por Sexo
    sexo_options = ["Todos", "Feminino", "Masculino"]
    sexo_filtro = st.sidebar.selectbox("Sexo", sexo_options)
    
    # Filtro por Faixa Etária
    idade_min = int(df['age'].min())
    idade_max = int(df['age'].max())
    idade_range = st.sidebar.slider(
        "Faixa Etária", 
        min_value=idade_min, 
        max_value=idade_max, 
        value=(idade_min, idade_max)
    )
    
    # Filtro por Condição Cardíaca
    condicao_options = ["Todos", "Sem Doença", "Com Doença"]
    condicao_filtro = st.sidebar.selectbox("Condição Cardíaca", condicao_options)
    
    # Filtro por Nível de Colesterol
    colesterol_min = int(df['chol'].min())
    colesterol_max = int(df['chol'].max())
    colesterol_range = st.sidebar.slider(
        "Nível de Colesterol", 
        min_value=colesterol_min, 
        max_value=colesterol_max, 
        value=(colesterol_min, colesterol_max)
    )
    
    # Filtro por Pressão Arterial
    pressao_min = int(df['trestbps'].min())
    pressao_max = int(df['trestbps'].max())
    pressao_range = st.sidebar.slider(
        "Pressão Arterial em Repouso", 
        min_value=pressao_min, 
        max_value=pressao_max, 
        value=(pressao_min, pressao_max)
    )
    
    return {
        'sexo': sexo_filtro,
        'idade_range': idade_range,
        'condicao': condicao_filtro,
        'colesterol_range': colesterol_range,
        'pressao_range': pressao_range
    }

# Função para aplicar filtros
def aplicar_filtros(df, filtros):
    df_filtrado = df.copy()
    
    # Filtro por Sexo
    if filtros['sexo'] == "Feminino":
        df_filtrado = df_filtrado[df_filtrado['sex'] == 0]
    elif filtros['sexo'] == "Masculino":
        df_filtrado = df_filtrado[df_filtrado['sex'] == 1]
    
    # Filtro por Faixa Etária
    df_filtrado = df_filtrado[
        (df_filtrado['age'] >= filtros['idade_range'][0]) & 
        (df_filtrado['age'] <= filtros['idade_range'][1])
    ]
    
    # Filtro por Condição Cardíaca
    if filtros['condicao'] == "Sem Doença":
        df_filtrado = df_filtrado[df_filtrado['target'] == 0]
    elif filtros['condicao'] == "Com Doença":
        df_filtrado = df_filtrado[df_filtrado['target'] == 1]
    
    # Filtro por Colesterol
    df_filtrado = df_filtrado[
        (df_filtrado['chol'] >= filtros['colesterol_range'][0]) & 
        (df_filtrado['chol'] <= filtros['colesterol_range'][1])
    ]
    
    # Filtro por Pressão Arterial
    df_filtrado = df_filtrado[
        (df_filtrado['trestbps'] >= filtros['pressao_range'][0]) & 
        (df_filtrado['trestbps'] <= filtros['pressao_range'][1])
    ]
    
    return df_filtrado

# =============================
# TRADUÇÃO DAS VARIÁVEIS PARA PORTUGUÊS
# =============================

traducao_variaveis = {
    'age': 'Idade',
    'sex': 'Sexo',
    'cp': 'Tipo de Dor no Peito',
    'trestbps': 'Pressão Arterial em Repouso',
    'chol': 'Colesterol',
    'fbs': 'Açúcar no Sangue em Jejum',
    'restecg': 'Resultado Eletrocardiográfico em Repouso',
    'thalach': 'Frequência Cardíaca Máxima',
    'exang': 'Angina Induzida por Exercício',
    'oldpeak': 'Depressão do Segmento ST',
    'slope': 'Inclinação do Segmento ST',
    'ca': 'Número de Vasos Principais',
    'thal': 'Thalassemia'
}

valores_sexo = {
    0: 'Feminino',
    1: 'Masculino'
}

valores_dor_peito = {
    0: 'Típica',
    1: 'Atípica', 
    2: 'Não anginosa',
    3: 'Assintomática'
}

valores_angina = {
    0: 'Não',
    1: 'Sim'
}

# =============================
# ABAS
# =============================

aba1, aba2, aba3 = st.tabs(["📊 Análise Descritiva", "🤖 Modelo Supervisionado", "🧬 Modelo Não Supervisionado"])

# =============================
# 1️⃣ ABA – ANÁLISE DESCRITIVA
# =============================

with aba1:
    st.header("Análise Descritiva dos Dados")
    
    # Criar e aplicar filtros
    filtros = criar_filtros_analise()
    df_filtrado = aplicar_filtros(df, filtros)
    
    # Mostrar estatísticas dos filtros
    st.sidebar.markdown("---")
    st.sidebar.metric("Pacientes Filtrados", len(df_filtrado))
    st.sidebar.metric("Total de Pacientes", len(df))
    
    if len(df_filtrado) == 0:
        st.warning("⚠️ Nenhum paciente encontrado com os filtros aplicados. Tente ajustar os critérios.")
        df_filtrado = df  # Usar dataset completo se filtro retornar vazio
    
    # Gráfico 1: Distribuição por Alvo (Pizza) - CORRIGIDO
    st.subheader("Distribuição por Condição Cardíaca")
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    counts = df_filtrado['target'].value_counts().sort_index()
    
    # CORREÇÃO: Ajustar labels dinamicamente baseado nos dados disponíveis
    labels = []
    valores = []
    cores = []
    
    if 0 in counts.index:
        labels.append('Sem doença')
        valores.append(counts[0])
        cores.append('skyblue')
    
    if 1 in counts.index:
        labels.append('Com doença cardíaca')
        valores.append(counts[1])
        cores.append('lightcoral')
    
    if len(valores) > 0:
        ax1.pie(valores, labels=labels, autopct='%1.1f%%', startangle=90, colors=cores)
        ax1.set_title("Distribuição por Condição Cardíaca")
        st.pyplot(fig1)
    else:
        st.info("Não há dados para exibir o gráfico de pizza com os filtros atuais.")
    
    # Gráfico 2: Distribuição por faixa etária
    st.subheader("Distribuição por Faixa Etária")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2_hist = sns.histplot(data=df_filtrado, x='age', kde=True, bins=20, ax=ax2)
    ax2.set_title('Distribuição por Faixa Etária')
    ax2.set_xlabel('Idade')
    ax2.set_ylabel('Contagem')
    
    for container in ax2_hist.containers:
        ax2.bar_label(container)
    
    st.pyplot(fig2)
    
    # Gráfico 3: Distribuição de gênero - CORRIGIDO
    st.subheader("Distribuição de Gênero")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    
    # CORREÇÃO: Ajustar cores dinamicamente baseado no filtro
    palette = []
    if filtros['sexo'] == "Todos":
        palette = ['red', 'blue']  # Feminino, Masculino
    elif filtros['sexo'] == "Feminino":
        palette = ['red']  # Apenas Feminino
    else:  # Masculino
        palette = ['blue']  # Apenas Masculino
    
    ax3_count = sns.countplot(data=df_filtrado, x='sex', palette=palette, ax=ax3)
    ax3.set_title('Distribuição de Gênero')
    ax3.set_xlabel('Gênero')
    ax3.set_ylabel('Contagem')
    
    # Ajustar labels do eixo X baseado nos dados disponíveis
    xtick_labels = []
    if 0 in df_filtrado['sex'].values:
        xtick_labels.append('Feminino')
    if 1 in df_filtrado['sex'].values:
        xtick_labels.append('Masculino')
    
    if len(xtick_labels) > 0:
        ax3.set_xticks(range(len(xtick_labels)))
        ax3.set_xticklabels(xtick_labels)
    
    for container in ax3_count.containers:
        ax3.bar_label(container)
    
    st.pyplot(fig3)
    
    # Gráfico 4: Distribuição por faixa etária e por gênero
    st.subheader("Distribuição por Faixa Etária e Gênero")
    
    if len(df_filtrado) > 0:
        # Calcula média, mediana e moda
        media = df_filtrado['age'].mean()
        mediana = df_filtrado['age'].median()
        moda = df_filtrado['age'].mode()[0] if len(df_filtrado['age'].mode()) > 0 else df_filtrado['age'].iloc[0]
        
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        ax4_hist = sns.histplot(data=df_filtrado, x='age', hue='sex', kde=True, bins=20, ax=ax4)
        
        # Linhas de média, mediana e moda
        ax4.axvline(media, color='red', linestyle='--', linewidth=2, label=f'Média: {media:.2f}')
        ax4.axvline(mediana, color='green', linestyle='--', linewidth=2, label=f'Mediana: {mediana:.2f}')
        ax4.axvline(moda, color='blue', linestyle='--', linewidth=2, label=f'Moda: {moda}')
        
        # Título e rótulos
        ax4.set_title('Distribuição por Faixa Etária e Gênero', fontsize=13)
        ax4.set_xlabel('Idade')
        ax4.set_ylabel('Contagem')
        
        # Legendas separadas
        handles, labels = ax4_hist.get_legend_handles_labels()
        if handles:
            ax4.legend(handles=handles[:2], labels=['Masculino', 'Feminino'], title='Gênero', loc='upper left')
        
        # Segunda legenda para média, mediana e moda
        from matplotlib.lines import Line2D
        legendas_linhas = [
            Line2D([0], [0], color='red', linestyle='--', linewidth=2, label=f'Média: {media:.2f}'),
            Line2D([0], [0], color='green', linestyle='--', linewidth=2, label=f'Mediana: {mediana:.2f}'),
            Line2D([0], [0], color='blue', linestyle='--', linewidth=2, label=f'Moda: {moda}')
        ]
        ax4.legend(handles=legendas_linhas, loc='upper right')
        
        # Adicionar a legenda de gênero novamente (pois foi sobrescrita)
        if handles:
            ax4.add_artist(ax4.get_legend())
            ax4.legend(handles=handles[:2], labels=['Masculino', 'Feminino'], title='Gênero', loc='upper left')
        
        # Exibir contagem nas barras
        for container in ax4_hist.containers:
            ax4.bar_label(container, fmt='%d', label_type='edge', fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig4)
    else:
        st.info("Não há dados suficientes para exibir este gráfico com os filtros atuais.")
    
    # Gráfico 5: Distribuição de Colesterol e Pressão Sanguínea
    st.subheader("Distribuição de Colesterol e Pressão Arterial")
    fig5, (ax5_1, ax5_2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Colesterol
    ax5_1_hist = sns.histplot(data=df_filtrado, x='chol', kde=True, bins=20, ax=ax5_1)
    ax5_1.set_title('Distribuição de Colesterol')
    ax5_1.set_xlabel('Colesterol (mg/dL)')
    ax5_1.set_ylabel('Contagem')
    for container in ax5_1_hist.containers:
        ax5_1.bar_label(container)
    
    # Pressão Sanguínea
    ax5_2_hist = sns.histplot(data=df_filtrado, x='trestbps', kde=True, bins=20, ax=ax5_2)
    ax5_2.set_title('Distribuição de Pressão Arterial em Repouso')
    ax5_2.set_xlabel('Pressão Arterial (mmHg)')
    ax5_2.set_ylabel('Contagem')
    for container in ax5_2_hist.containers:
        ax5_2.bar_label(container)
    
    plt.tight_layout()
    st.pyplot(fig5)
    
    # Gráfico 6: Colesterol por Condição
    st.subheader("Colesterol por Condição de Saúde")
    if len(df_filtrado['target'].unique()) > 1:
        fig6, ax6 = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=df_filtrado, x='target', y='chol', ax=ax6)
        ax6.set_title('Colesterol por Condição Cardíaca')
        ax6.set_xlabel('Condição')
        ax6.set_ylabel('Colesterol (mg/dL)')
        ax6.set_xticks([0, 1])
        ax6.set_xticklabels(['Sem Doença', 'Com Doença'])
        st.pyplot(fig6)
    else:
        st.info("Não há dados suficientes para comparar colesterol entre condições com os filtros atuais.")
    
    # Gráfico 7: Pressão Arterial por Condição
    st.subheader("Pressão Arterial por Condição de Saúde")
    if len(df_filtrado['target'].unique()) > 1:
        fig7, ax7 = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=df_filtrado, x='target', y='trestbps', ax=ax7)
        ax7.set_title('Pressão Arterial por Condição Cardíaca')
        ax7.set_xlabel('Condição')
        ax7.set_ylabel('Pressão Arterial (mmHg)')
        ax7.set_xticks([0, 1])
        ax7.set_xticklabels(['Sem Doença', 'Com Doença'])
        st.pyplot(fig7)
    else:
        st.info("Não há dados suficientes para comparar pressão arterial entre condições com os filtros atuais.")
    
    # Gráfico 8: Dispersão Idade vs Colesterol
    st.subheader("Relação entre Idade e Colesterol")
    fig8, ax8 = plt.subplots(figsize=(8, 5))
    targets = df_filtrado['target'].unique()
    if len(targets) > 0:
        for t in targets:
            subset = df_filtrado[df_filtrado['target'] == t]
            ax8.scatter(subset['age'], subset['chol'], label=f"{'Com Doença' if t == 1 else 'Sem Doença'}", alpha=0.7)
        ax8.set_xlabel("Idade")
        ax8.set_ylabel("Colesterol (mg/dL)")
        ax8.set_title("Relação entre Idade e Colesterol")
        ax8.legend()
        st.pyplot(fig8)
    else:
        st.info("Não há dados suficientes para exibir este gráfico com os filtros atuais.")

# =============================
# 2️⃣ ABA – MODELO SUPERVISIONADO
# =============================

with aba2:
    st.header("Modelo Supervisionado – Previsão de Risco de Doença Cardíaca")

    # Separação das variáveis
    X = df.drop('target', axis=1)
    y = df['target']

    # Normalização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Treinamento do modelo
    modelo = RandomForestClassifier(n_estimators=200, random_state=42)
    modelo.fit(X_train, y_train)

    # Entrada manual do usuário
    st.subheader("Informe os dados do paciente:")
    
    # Criar colunas para organizar os inputs
    col1, col2 = st.columns(2)
    
    entrada = {}
    with col1:
        entrada['age'] = st.number_input(
            "Idade", 
            min_value=20, 
            max_value=100, 
            value=54,
            help="Idade do paciente em anos"
        )
        
        entrada['sex'] = st.selectbox(
            "Sexo",
            options=[0, 1],
            format_func=lambda x: "Feminino" if x == 0 else "Masculino",
            help="Sexo biológico do paciente"
        )
        
        entrada['cp'] = st.selectbox(
            "Tipo de Dor no Peito",
            options=[0, 1, 2, 3],
            format_func=lambda x: ["Típica", "Atípica", "Não anginosa", "Assintomática"][x],
            help="Tipo de dor no peito relatada"
        )
        
        entrada['trestbps'] = st.number_input(
            "Pressão Arterial em Repouso (mmHg)",
            min_value=80,
            max_value=200,
            value=130,
            help="Pressão arterial sistólica em repouso"
        )
        
        entrada['chol'] = st.number_input(
            "Colesterol (mg/dL)",
            min_value=100,
            max_value=600,
            value=240,
            help="Nível de colesterol sérico"
        )
        
        entrada['fbs'] = st.selectbox(
            "Açúcar no Sangue em Jejum > 120 mg/dL",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Açúcar no sangue em jejum elevado"
        )
    
    with col2:
        entrada['restecg'] = st.selectbox(
            "Resultado Eletrocardiográfico em Repouso",
            options=[0, 1, 2],
            format_func=lambda x: ["Normal", "Anormalidade ST-T", "Hipertrofia ventricular"][x],
            help="Resultado do eletrocardiograma em repouso"
        )
        
        entrada['thalach'] = st.number_input(
            "Frequência Cardíaca Máxima",
            min_value=60,
            max_value=220,
            value=150,
            help="Frequência cardíaca máxima alcançada"
        )
        
        entrada['exang'] = st.selectbox(
            "Angina Induzida por Exercício",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Presença de angina induzida por exercício"
        )
        
        entrada['oldpeak'] = st.number_input(
            "Depressão do Segmento ST",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Depressão do segmento ST induzida por exercício"
        )
        
        entrada['slope'] = st.selectbox(
            "Inclinação do Segmento ST",
            options=[0, 1, 2],
            format_func=lambda x: ["Ascendente", "Plana", "Descendente"][x],
            help="Inclinação do segmento ST no pico do exercício"
        )
        
        entrada['ca'] = st.selectbox(
            "Número de Vasos Principais",
            options=[0, 1, 2, 3],
            help="Número de vasos principais coloridos por fluoroscopia"
        )
        
        entrada['thal'] = st.selectbox(
            "Thalassemia",
            options=[0, 1, 2, 3],
            format_func=lambda x: ["Normal", "Defeito Fixo", "Defeito Reversível", "Não disponível"][x],
            help="Resultado do teste de thalassemia"
        )

    entrada_df = pd.DataFrame([entrada])
    entrada_scaled = scaler.transform(entrada_df)

    # Predição
    if st.button("Realizar Previsão", type="primary"):
        pred_prob = modelo.predict_proba(entrada_scaled)[0][1]
        pred_class = modelo.predict(entrada_scaled)[0]

        st.subheader("Resultado da Previsão:")
        if pred_class == 1:
            st.error(f"⚠️ Risco Elevado de Doença Cardíaca ({pred_prob*100:.1f}% de probabilidade)")
            st.warning("Recomenda-se consulta com cardiologista para avaliação detalhada.")
        else:
            st.success(f"✅ Baixo Risco de Doença Cardíaca ({(1 - pred_prob)*100:.1f}% de probabilidade)")
            st.info("Continue com hábitos saudáveis e check-ups regulares.")

        # Exibe acurácia
        st.info(f"📊 Acurácia do modelo (Random Forest): {modelo.score(X_test, y_test)*100:.2f}%")

# =============================
# 3️⃣ ABA – MODELO NÃO SUPERVISIONADO
# =============================

with aba3:
    st.header("Modelo Não Supervisionado – Agrupamento de Pacientes")
    st.markdown("""
    O algoritmo **K-Means** agrupa pacientes com base em características clínicas semelhantes.
    """)

    # Aplicando KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df['cluster'] = clusters

    # Análise Interativa dos Clusters
    st.subheader("Análise Interativa")
    
    # Mapeamento dos clusters para perfis descritivos
    # Primeiro vamos analisar qual cluster corresponde a qual perfil
    analise_risco = df.groupby('cluster').agg({
        'target': 'mean',
        'age': 'mean',
        'chol': 'mean'
    }).round(3)
    
    # Ordenar clusters por taxa de doença (do menor para o maior risco)
    analise_risco = analise_risco.sort_values('target')
    
    # Mapeamento automático baseado na taxa de doença
    mapeamento_clusters = {}
    perfis = ["Perfil de Baixo Risco", "Perfil de Risco Intermediário", "Perfil de Alto Risco"]
    
    for i, (cluster_idx, row) in enumerate(analise_risco.iterrows()):
        mapeamento_clusters[cluster_idx] = perfis[i]
    
    # Criar mapeamento reverso para facilitar
    mapeamento_reverso = {v: k for k, v in mapeamento_clusters.items()}
    
    # Seleção do cluster para análise - AGORA COM PERFIS DESCRITIVOS
    perfil_selecionado = st.selectbox(
        "Selecione o Perfil para Análise Detalhada:",
        options=perfis,
        help="Escolha o perfil de risco para visualizar características e recomendações específicas"
    )
    
    # Obter o cluster numérico correspondente ao perfil selecionado
    cluster_selecionado = mapeamento_reverso[perfil_selecionado]
    
    # Filtrar dados do cluster selecionado
    cluster_data = df[df['cluster'] == cluster_selecionado]
    
    # Exibir badge do perfil selecionado
    if "Alto Risco" in perfil_selecionado:
        st.error(f"🔴 **{perfil_selecionado}**")
    elif "Intermediário" in perfil_selecionado:
        st.warning(f"🟡 **{perfil_selecionado}**")
    else:
        st.success(f"🟢 **{perfil_selecionado}**")
    
    # Estatísticas do cluster
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            f"Pacientes no {perfil_selecionado.split()[-1]}", 
            len(cluster_data)
        )
    
    with col2:
        percentual_cluster = (len(cluster_data) / len(df)) * 100
        st.metric(
            "Percentual do Total",
            f"{percentual_cluster:.1f}%"
        )
    
    with col3:
        taxa_doenca_cluster = (cluster_data['target'].sum() / len(cluster_data)) * 100
        st.metric(
            "Taxa de Doença Cardíaca",
            f"{taxa_doenca_cluster:.1f}%"
        )
    
    # Características do perfil selecionado
    st.subheader(f"Características do {perfil_selecionado}")
    
    # Métricas principais do cluster
    st.write("**Principais Características:**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        idade_media = cluster_data['age'].mean()
        st.metric("Idade Média", f"{idade_media:.1f} anos")
    
    with col2:
        colesterol_medio = cluster_data['chol'].mean()
        st.metric("Colesterol Médio", f"{colesterol_medio:.1f} mg/dL")
    
    with col3:
        pressao_media = cluster_data['trestbps'].mean()
        st.metric("Pressão Arterial Média", f"{pressao_media:.1f} mmHg")
    
    with col4:
        freq_card_max_media = cluster_data['thalach'].mean()
        st.metric("Freq. Cardíaca Máx. Média", f"{freq_card_max_media:.1f} bpm")
    
    # Distribuição de gênero no cluster
    st.write("**Distribuição por Gênero:**")
    genero_cluster = cluster_data['sex'].value_counts()
    
    if len(genero_cluster) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            if 0 in genero_cluster.index:
                st.metric("Feminino", genero_cluster[0])
            else:
                st.metric("Feminino", 0)
        
        with col2:
            if 1 in genero_cluster.index:
                st.metric("Masculino", genero_cluster[1])
            else:
                st.metric("Masculino", 0)
    
    # Recomendações baseadas no perfil
    st.subheader("Recomendações")
    
    # Análise de risco do cluster
    risco_cluster = "ALTO" if taxa_doenca_cluster > 50 else "MODERADO" if taxa_doenca_cluster > 25 else "BAIXO"
    
    st.write(f"**Nível de Risco:** {risco_cluster}")
    
    # Recomendações específicas baseadas no perfil selecionado
    if perfil_selecionado == "Perfil de Alto Risco":
        st.error("""
        **🔴 Plano de Ação - Alto Risco:**
        
        **Ações Imediatas:**
        - Consulta cardiológica URGENTE
        - Monitoramento diário da pressão arterial
        - Exames de sangue mensais
        - Avaliação de necessidade de medicação
        
        **Intervenções:**
        - Dieta rigorosa (baixo sódio, baixa gordura)
        - Exercícios supervisionados
        - Controle rigoroso de peso
        - Abandono do tabagismo (se aplicável)
        - Limitação de álcool
        
        **Acompanhamento:**
        - Consultas trimestrais
        - Eletrocardiograma semestral
        - Teste de esforço anual
        """)
    elif perfil_selecionado == "Perfil de Risco Intermediário":
        st.warning("""
        **🟡 Plano de Ação - Risco Intermediário:**
        
        **Ações Recomendadas:**
        - Consulta cardiológica em 30 dias
        - Monitoramento semanal da pressão
        - Exames de sangue trimestrais
        
        **Intervenções:**
        - Dieta mediterrânea
        - Exercícios regulares (150min/semana)
        - Controle de peso
        - Redução de estresse
        
        **Acompanhamento:**
        - Consultas semestrais
        - Check-up cardiovascular anual
        - Educação em saúde
        """)
    else:  # Baixo Risco
        st.success("""
        **🟢 Plano de Ação - Baixo Risco:**
        
        **Ações de Manutenção:**
        - Check-up anual com clínico geral
        - Monitoramento preventivo
        - Manutenção do estilo de vida saudável
        
        **Prevenção:**
        - Alimentação balanceada
        - Atividade física regular
        - Controle do estresse
        - Sono de qualidade
        
        **Acompanhamento:**
        - Consultas anuais de rotina
        - Exames preventivos
        - Educação continuada em saúde
        """)
    
    # Comparação entre todos os perfis
    st.subheader("Comparação entre Todos os Perfis")
    
    # Criar dataframe comparativo com nomes descritivos
    comparacao = df.groupby('cluster').agg({
        'age': 'mean',
        'chol': 'mean', 
        'trestbps': 'mean',
        'thalach': 'mean',
        'target': 'mean'
    }).round(2)
    
    # Renomear os índices para os perfis descritivos
    comparacao.index = [mapeamento_clusters[idx] for idx in comparacao.index]
    comparacao.columns = ['Idade Média', 'Colesterol Médio', 'Pressão Média', 
                         'Freq. Card. Máx. Média', 'Taxa de Doença']
    
    st.dataframe(comparacao)
    

    
