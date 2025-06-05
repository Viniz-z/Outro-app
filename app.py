import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="🎾 Tennis Match Predictor",
    page_icon="🎾",
    layout="wide"
)

st.title("🎾 Preditor de Partidas de Tênis")
st.markdown("### Carregue seus dados históricos e faça predições!")
st.markdown("---")

# Função para processar dados carregados
def process_match_data(df):
    """Processa os dados de partidas carregados"""
    required_columns = ['jogador1', 'jogador2', 'vencedor', 'superficie', 'data']
    
    # Verificar se todas as colunas necessárias existem
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Colunas obrigatórias ausentes: {missing_cols}")
        st.info("Colunas necessárias: jogador1, jogador2, vencedor, superficie, data")
        return None
    
    # Limpar e processar dados
    df = df.dropna(subset=required_columns)
    df['data'] = pd.to_datetime(df['data'], errors='coerce')
    df = df.dropna(subset=['data'])
    
    return df

# Função para treinar modelo com dados do usuário
@st.cache_data
def train_model_with_user_data(df):
    """Treina modelo com dados fornecidos pelo usuário"""
    try:
        # Preparar encoders
        le_player = LabelEncoder()
        le_surface = LabelEncoder()
        
        # Obter todos os jogadores únicos
        all_players = list(set(df['jogador1'].tolist() + df['jogador2'].tolist()))
        le_player.fit(all_players)
        le_surface.fit(df['superficie'].unique())
        
        # Criar features e targets
        features = []
        targets = []
        
        for _, row in df.iterrows():
            # Calcular estatísticas do jogador
            j1_wins = len(df[(df['vencedor'] == row['jogador1'])])
            j1_total = len(df[(df['jogador1'] == row['jogador1']) | (df['jogador2'] == row['jogador1'])])
            j1_winrate = j1_wins / j1_total if j1_total > 0 else 0.5
            
            j2_wins = len(df[(df['vencedor'] == row['jogador2'])])
            j2_total = len(df[(df['jogador1'] == row['jogador2']) | (df['jogador2'] == row['jogador2'])])
            j2_winrate = j2_wins / j2_total if j2_total > 0 else 0.5
            
            # Confronto direto
            confrontos = df[((df['jogador1'] == row['jogador1']) & (df['jogador2'] == row['jogador2'])) |
                           ((df['jogador1'] == row['jogador2']) & (df['jogador2'] == row['jogador1']))]
            j1_wins_h2h = len(confrontos[confrontos['vencedor'] == row['jogador1']])
            total_h2h = len(confrontos)
            h2h_advantage = j1_wins_h2h / total_h2h if total_h2h > 0 else 0.5
            
            feature_row = [
                le_player.transform([row['jogador1']])[0],
                le_player.transform([row['jogador2']])[0],
                j1_winrate,
                j2_winrate,
                le_surface.transform([row['superficie']])[0],
                h2h_advantage,
                total_h2h
            ]
            
            features.append(feature_row)
            targets.append(1 if row['vencedor'] == row['jogador1'] else 0)
        
        # Treinar modelo
        if len(features) > 10:  # Mínimo de dados para treinar
            X = np.array(features)
            y = np.array(targets)
            
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            return model, le_player, le_surface, all_players
        else:
            st.error("Dados insuficientes para treinar o modelo (mínimo 10 partidas)")
            return None, None, None, None
            
    except Exception as e:
        st.error(f"Erro ao treinar modelo: {str(e)}")
        return None, None, None, None

# Interface principal
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📁 Carregamento de Dados")
    
    # Upload de arquivo
    uploaded_file = st.file_uploader(
        "Carregue seu arquivo CSV com histórico de partidas",
        type=['csv'],
        help="Arquivo deve conter: jogador1, jogador2, vencedor, superficie, data"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Arquivo carregado: {len(df)} registros")
            
            # Mostrar preview dos dados
            st.subheader("👀 Preview dos Dados")
            st.dataframe(df.head(), use_container_width=True)
            
            # Processar dados
            processed_df = process_match_data(df)
            
            if processed_df is not None:
                st.success("✅ Dados processados com sucesso!")
                
                # Estatísticas básicas
                st.subheader("📊 Estatísticas dos Dados")
                col_stats1, col_stats2 = st.columns(2)
                
                with col_stats1:
                    st.metric("Total de Partidas", len(processed_df))
                    st.metric("Jogadores Únicos", len(set(processed_df['jogador1'].tolist() + processed_df['jogador2'].tolist())))
                
                with col_stats2:
                    st.metric("Superfícies", processed_df['superficie'].nunique())
                    st.metric("Período", f"{processed_df['data'].min().strftime('%Y')} - {processed_df['data'].max().strftime('%Y')}")
                
                # Treinar modelo
                model, le_player, le_surface, all_players = train_model_with_user_data(processed_df)
                
                if model is not None:
                    st.success("🤖 Modelo treinado com sucesso!")
                    
                    # Salvar no session state
                    st.session_state['model'] = model
                    st.session_state['le_player'] = le_player
                    st.session_state['le_surface'] = le_surface
                    st.session_state['all_players'] = all_players
                    st.session_state['df'] = processed_df
        
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {str(e)}")

with col2:
    st.subheader("🔮 Fazer Predição")
    
    # Verificar se modelo está disponível
    if 'model' in st.session_state:
        model = st.session_state['model']
        le_player = st.session_state['le_player']
        le_surface = st.session_state['le_surface']
        all_players = st.session_state['all_players']
        df = st.session_state['df']
        
        # Inputs para predição
        jogador1 = st.selectbox("Jogador 1", sorted(all_players))
        jogador2 = st.selectbox("Jogador 2", [p for p in sorted(all_players) if p != jogador1])
        superficie = st.selectbox("Superfície", sorted(df['superficie'].unique()))
        
        if st.button("🎯 Fazer Predição", type="primary"):
            try:
                # Calcular estatísticas para predição
                j1_wins = len(df[df['vencedor'] == jogador1])
                j1_total = len(df[(df['jogador1'] == jogador1) | (df['jogador2'] == jogador1)])
                j1_winrate = j1_wins / j1_total if j1_total > 0 else 0.5
                
                j2_wins = len(df[df['vencedor'] == jogador2])
                j2_total = len(df[(df['jogador1'] == jogador2) | (df['jogador2'] == jogador2)])
                j2_winrate = j2_wins / j2_total if j2_total > 0 else 0.5
                
                # Confronto direto
                confrontos = df[((df['jogador1'] == jogador1) & (df['jogador2'] == jogador2)) |
                               ((df['jogador1'] == jogador2) & (df['jogador2'] == jogador1))]
                j1_wins_h2h = len(confrontos[confrontos['vencedor'] == jogador1])
                total_h2h = len(confrontos)
                h2h_advantage = j1_wins_h2h / total_h2h if total_h2h > 0 else 0.5
                
                # Preparar features
                feature_row = [
                    le_player.transform([jogador1])[0],
                    le_player.transform([jogador2])[0],
                    j1_winrate,
                    j2_winrate,
                    le_surface.transform([superficie])[0],
                    h2h_advantage,
                    total_h2h
                ]
                
                # Fazer predição
                prob = model.predict_proba([feature_row])[0]
                prediction = model.predict([feature_row])[0]
                
                # Mostrar resultados
                winner = jogador1 if prediction == 1 else jogador2
                confidence = max(prob) * 100
                
                st.success(f"🏆 **Vencedor Previsto: {winner}**")
                st.info(f"📊 **Confiança: {confidence:.1f}%**")
                
                # Gráfico de probabilidades
                fig = go.Figure(data=[
                    go.Bar(x=[jogador1, jogador2], 
                          y=[prob[1]*100, prob[0]*100],
                          marker_color=['#FF6B6B', '#4ECDC4'],
                          text=[f'{prob[1]*100:.1f}%', f'{prob[0]*100:.1f}%'],
                          textposition='auto')
                ])
                fig.update_layout(
                    title="Probabilidade de Vitória",
                    yaxis_title="Probabilidade (%)",
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar estatísticas dos jogadores
                st.subheader("📈 Estatísticas dos Jogadores")
                col_j1, col_j2 = st.columns(2)
                
                with col_j1:
                    st.metric(f"{jogador1} - Taxa de Vitória", f"{j1_winrate:.1%}")
                    st.metric(f"{jogador1} - Total de Jogos", j1_total)
                
                with col_j2:
                    st.metric(f"{jogador2} - Taxa de Vitória", f"{j2_winrate:.1%}")
                    st.metric(f"{jogador2} - Total de Jogos", j2_total)
                
                if total_h2h > 0:
                    st.metric("Confrontos Diretos", f"{j1_wins_h2h}-{total_h2h-j1_wins_h2h} (Total: {total_h2h})")
                else:
                    st.info("Nenhum confronto direto encontrado nos dados")
                
            except Exception as e:
                st.error(f"Erro na predição: {str(e)}")
    else:
        st.info("👆 Carregue seus dados primeiro para fazer predições")

# Seção de análise (só aparece se dados carregados)
if 'df' in st.session_state:
    st.markdown("---")
    st.subheader("📊 Análise dos Seus Dados")
    
    df = st.session_state['df']
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Vitórias por superfície
        surface_data = df['superficie'].value_counts()
        fig1 = px.pie(values=surface_data.values, names=surface_data.index,
                      title="Distribuição de Partidas por Superfície")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col4:
        # Top jogadores por vitórias
        winner_counts = df['vencedor'].value_counts().head(10)
        fig2 = px.bar(x=winner_counts.index, y=winner_counts.values,
                      title="Top 10 Jogadores (Vitórias)")
        fig2.update_layout(xaxis_title="Jogador", yaxis_title="Vitórias")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Tabela de partidas recentes
    st.subheader("🕐 Partidas Recentes")
    recent_matches = df.sort_values('data', ascending=False).head(10)
    display_df = recent_matches[['jogador1', 'jogador2', 'vencedor', 'superficie', 'data']].copy()
    display_df['data'] = display_df['data'].dt.strftime('%d/%m/%Y')
    st.dataframe(display_df, use_container_width=True)

# Exemplo de formato de dados
st.markdown("---")
st.subheader("📝 Formato dos Dados")
st.markdown("Seu arquivo CSV deve ter as seguintes colunas:")

example_data = {
    'jogador1': ['Novak Djokovic', 'Rafael Nadal', 'Roger Federer'],
    'jogador2': ['Rafael Nadal', 'Andy Murray', 'Novak Djokovic'],
    'vencedor': ['Novak Djokovic', 'Rafael Nadal', 'Roger Federer'],
    'superficie': ['Hard', 'Clay', 'Grass'],
    'data': ['2023-01-15', '2023-02-20', '2023-03-10']
}

example_df = pd.DataFrame(example_data)
st.dataframe(example_df, use_container_width=True)

st.markdown("""
**Colunas obrigatórias:**
- `jogador1`: Nome do primeiro jogador
- `jogador2`: Nome do segundo jogador  
- `vencedor`: Nome do jogador que venceu a partida
- `superficie`: Tipo de quadra (Clay, Hard, Grass, etc.)
- `data`: Data da partida (formato: YYYY-MM-DD)

**Colunas opcionais:**
- `torneio`: Nome do torneio
- `round`: Fase do torneio
- `sets`: Placar em sets
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🎾 <strong>Tennis Match Predictor</strong> - Versão Offline</p>
    <p><em>Carregue seus dados e faça predições personalizadas</em></p>
</div>
""", unsafe_allow_html=True)
