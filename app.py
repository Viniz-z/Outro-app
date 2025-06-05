import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import io

# Configuração da página
st.set_page_config(
    page_title="Análise de Tênis - Dashboard",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2e8b57;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-container {
        background: linear-gradient(90deg, #2e8b57 0%, #20b2aa 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .player-stats {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem;
    }
    .winner-highlight {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🎾 Dashboard de Análise de Tênis</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controles")
        
        # Upload de arquivo
        uploaded_file = st.file_uploader(
            "Carregar histórico de games (CSV)",
            type=['csv'],
            help="CSV deve conter: game_number, player1_score, player2_score, winner, set_number"
        )
        
        # Opções de análise
        analysis_type = st.selectbox(
            "Tipo de Análise",
            ["Resumo da Partida", "Análise por Sets", "Momentum", "Estatísticas Avançadas"]
        )
        
        # Configurações de gráficos
        st.subheader("🎨 Configurações Visuais")
        color_theme = st.selectbox(
            "Tema de Cores",
            ["plotly", "viridis", "plasma", "inferno", "magma"]
        )
        
        chart_height = st.slider("Altura dos Gráficos", 300, 800, 500)
    
    # Área principal
    if uploaded_file is not None:
        try:
            df = load_tennis_data(uploaded_file)
            
            if df is not None:
                st.success(f"✅ Partida carregada! {len(df)} games analisados.")
                
                # Análise baseada na seleção
                if analysis_type == "Resumo da Partida":
                    show_match_summary(df)
                elif analysis_type == "Análise por Sets":
                    show_set_analysis(df, color_theme, chart_height)
                elif analysis_type == "Momentum":
                    show_momentum_analysis(df, color_theme, chart_height)
                elif analysis_type == "Estatísticas Avançadas":
                    show_advanced_stats(df, color_theme, chart_height)
                    
        except Exception as e:
            st.error(f"❌ Erro ao processar o arquivo: {str(e)}")
            st.info("💡 Certifique-se de que o CSV contém as colunas necessárias.")
    
    else:
        show_sample_tennis_data()

def load_tennis_data(uploaded_file):
    """Carrega e valida dados de tênis"""
    df = pd.read_csv(uploaded_file)
    
    # Colunas esperadas
    required_cols = ['game_number', 'player1_score', 'player2_score', 'winner']
    
    # Verificar se as colunas necessárias existem
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Colunas faltando: {', '.join(missing_cols)}")
        st.info("📋 Colunas esperadas: game_number, player1_score, player2_score, winner, set_number (opcional)")
        return None
    
    # Adicionar set_number se não existir
    if 'set_number' not in df.columns:
        df['set_number'] = 1
    
    # Adicionar colunas calculadas
    df['score_diff'] = df['player1_score'] - df['player2_score']
    df['total_points'] = df['player1_score'] + df['player2_score']
    df['player1_win'] = (df['winner'] == 1).astype(int)
    df['player2_win'] = (df['winner'] == 2).astype(int)
    
    return df

def show_match_summary(df):
    """Mostra resumo geral da partida"""
    st.header("🏆 Resumo da Partida")
    
    # Estatísticas gerais
    total_games = len(df)
    player1_games = df['player1_win'].sum()
    player2_games = df['player2_win'].sum()
    total_points_p1 = df['player1_score'].sum()
    total_points_p2 = df['player2_score'].sum()
    
    # Determinar vencedor
    if player1_games > player2_games:
        winner_text = "🏆 Jogador 1 Venceu!"
        winner_color = "#56ab2f"
    elif player2_games > player1_games:
        winner_text = "🏆 Jogador 2 Venceu!"
        winner_color = "#2e8b57"
    else:
        winner_text = "🤝 Empate!"
        winner_color = "#ffa500"
    
    st.markdown(f"""
    <div class="winner-highlight" style="background: linear-gradient(135deg, {winner_color} 0%, #a8e6cf 100%);">
        <h2>{winner_text}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3>{total_games}</h3>
            <p>Total de Games</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h3>{player1_games} - {player2_games}</h3>
            <p>Games Vencidos</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <h3>{total_points_p1} - {total_points_p2}</h3>
            <p>Total de Pontos</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_game_length = df['total_points'].mean()
        st.markdown(f"""
        <div class="metric-container">
            <h3>{avg_game_length:.1f}</h3>
            <p>Pontos/Game Médio</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Gráfico de barras comparativo
    col1, col2 = st.columns(2)
    
    with col1:
        # Games por jogador
        fig_games = go.Figure(data=[
            go.Bar(name='Jogador 1', x=['Games Vencidos'], y=[player1_games], marker_color='#2e8b57'),
            go.Bar(name='Jogador 2', x=['Games Vencidos'], y=[player2_games], marker_color='#20b2aa')
        ])
        fig_games.update_layout(title="Games Vencidos por Jogador", barmode='group')
        st.plotly_chart(fig_games, use_container_width=True)
    
    with col2:
        # Pontos por jogador
        fig_points = go.Figure(data=[
            go.Bar(name='Jogador 1', x=['Total de Pontos'], y=[total_points_p1], marker_color='#2e8b57'),
            go.Bar(name='Jogador 2', x=['Total de Pontos'], y=[total_points_p2], marker_color='#20b2aa')
        ])
        fig_points.update_layout(title="Total de Pontos por Jogador", barmode='group')
        st.plotly_chart(fig_points, use_container_width=True)
    
    # Tabela de games
    st.subheader("📊 Histórico de Games")
    display_df = df[['game_number', 'set_number', 'player1_score', 'player2_score', 'winner', 'total_points']].copy()
    display_df.columns = ['Game', 'Set', 'Jogador 1', 'Jogador 2', 'Vencedor', 'Total Pontos']
    st.dataframe(display_df, use_container_width=True)

def show_set_analysis(df, color_theme, chart_height):
    """Análise detalhada por sets"""
    st.header("📈 Análise por Sets")
    
    sets = df['set_number'].unique()
    
    for set_num in sorted(sets):
        set_data = df[df['set_number'] == set_num]
        
        st.subheader(f"Set {set_num}")
        
        col1, col2, col3 = st.columns(3)
        
        p1_games = set_data['player1_win'].sum()
        p2_games = set_data['player2_win'].sum()
        total_games = len(set_data)
        
        with col1:
            st.metric("Jogador 1", f"{p1_games} games")
        with col2:
            st.metric("Jogador 2", f"{p2_games} games")
        with col3:
            st.metric("Total Games", total_games)
        
        # Gráfico de evolução do set
        fig = go.Figure()
        
        cumsum_p1 = set_data['player1_win'].cumsum()
        cumsum_p2 = set_data['player2_win'].cumsum()
        
        fig.add_trace(go.Scatter(
            x=set_data['game_number'],
            y=cumsum_p1,
            mode='lines+markers',
            name='Jogador 1',
            line=dict(color='#2e8b57', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=set_data['game_number'],
            y=cumsum_p2,
            mode='lines+markers',
            name='Jogador 2',
            line=dict(color='#20b2aa', width=3)
        ))
        
        fig.update_layout(
            title=f"Evolução do Set {set_num}",
            xaxis_title="Número do Game",
            yaxis_title="Games Acumulados",
            height=chart_height//2
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_momentum_analysis(df, color_theme, chart_height):
    """Análise de momentum durante a partida"""
    st.header("⚡ Análise de Momentum")
    
    # Calcular momentum (sequências de vitórias)
    df_momentum = df.copy()
    
    # Momentum = diferença cumulativa de games
    df_momentum['momentum'] = (df_momentum['player1_win'] - df_momentum['player2_win']).cumsum()
    
    # Gráfico principal de momentum
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_momentum['game_number'],
        y=df_momentum['momentum'],
        mode='lines+markers',
        name='Momentum',
        line=dict(color='#ff6b6b', width=3),
        fill='tonexty',
        fillcolor='rgba(255, 107, 107, 0.1)'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Equilíbrio")
    
    fig.update_layout(
        title="Momentum da Partida (Positivo = Jogador 1, Negativo = Jogador 2)",
        xaxis_title="Número do Game",
        yaxis_title="Diferença de Games",
        height=chart_height
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Estatísticas de momentum
    col1, col2, col3 = st.columns(3)
    
    max_momentum_p1 = df_momentum['momentum'].max()
    max_momentum_p2 = abs(df_momentum['momentum'].min())
    momentum_changes = len(df_momentum[df_momentum['momentum'].diff().abs() > 0])
    
    with col1:
        st.metric("Maior Vantagem J1", f"+{max_momentum_p1}")
    with col2:
        st.metric("Maior Vantagem J2", f"+{max_momentum_p2}")
    with col3:
        st.metric("Mudanças de Momentum", momentum_changes)
    
    # Heatmap de pontos por game
    st.subheader("🔥 Intensidade dos Games")
    
    fig_heat = px.bar(
        df,
        x='game_number',
        y='total_points',
        color='total_points',
        color_continuous_scale=color_theme,
        title="Pontos por Game (Intensidade)",
        labels={'total_points': 'Total de Pontos', 'game_number': 'Número do Game'}
    )
    
    fig_heat.update_layout(height=chart_height//2)
    st.plotly_chart(fig_heat, use_container_width=True)

def show_advanced_stats(df, color_theme, chart_height):
    """Estatísticas avançadas"""
    st.header("📊 Estatísticas Avançadas")
    
    # Eficiência por jogador
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Eficiência do Jogador 1")
        p1_efficiency = (df['player1_score'].sum() / df['total_points'].sum()) * 100
        p1_game_win_rate = (df['player1_win'].sum() / len(df)) * 100
        p1_avg_points_per_game = df['player1_score'].mean()
        
        st.metric("% de Pontos Vencidos", f"{p1_efficiency:.1f}%")
        st.metric("% de Games Vencidos", f"{p1_game_win_rate:.1f}%")
        st.metric("Pontos/Game Médio", f"{p1_avg_points_per_game:.1f}")
    
    with col2:
        st.subheader("🎯 Eficiência do Jogador 2")
        p2_efficiency = (df['player2_score'].sum() / df['total_points'].sum()) * 100
        p2_game_win_rate = (df['player2_win'].sum() / len(df)) * 100
        p2_avg_points_per_game = df['player2_score'].mean()
        
        st.metric("% de Pontos Vencidos", f"{p2_efficiency:.1f}%")
        st.metric("% de Games Vencidos", f"{p2_game_win_rate:.1f}%")
        st.metric("Pontos/Game Médio", f"{p2_avg_points_per_game:.1f}")
    
    # Análise de break points (games longos)
    st.subheader("🔥 Análise de Games Críticos")
    
    # Games com mais de X pontos são considerados críticos
    critical_threshold = df['total_points'].quantile(0.75)
    critical_games = df[df['total_points'] >= critical_threshold]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Games Críticos", len(critical_games))
    with col2:
        p1_critical_wins = critical_games['player1_win'].sum()
        st.metric("J1 em Games Críticos", f"{p1_critical_wins}/{len(critical_games)}")
    with col3:
        p2_critical_wins = critical_games['player2_win'].sum()
        st.metric("J2 em Games Críticos", f"{p2_critical_wins}/{len(critical_games)}")
    
    # Distribuição de pontos por game
    fig_dist = px.histogram(
        df,
        x='total_points',
        nbins=20,
        title="Distribuição da Duração dos Games",
        color_discrete_sequence=['#2e8b57']
    )
    fig_dist.update_layout(
        xaxis_title="Total de Pontos no Game",
        yaxis_title="Frequência",
        height=chart_height//2
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Padrões de vitória
    st.subheader("📈 Padrões de Performance")
    
    # Criar rolling average
    window = 5
    df_patterns = df.copy()
    df_patterns['p1_rolling_avg'] = df_patterns['player1_win'].rolling(window=window).mean()
    df_patterns['p2_rolling_avg'] = df_patterns['player2_win'].rolling(window=window).mean()
    
    fig_patterns = go.Figure()
    
    fig_patterns.add_trace(go.Scatter(
        x=df_patterns['game_number'],
        y=df_patterns['p1_rolling_avg'],
        mode='lines',
        name=f'Jogador 1 (Média móvel {window} games)',
        line=dict(color='#2e8b57', width=2)
    ))
    
    fig_patterns.add_trace(go.Scatter(
        x=df_patterns['game_number'],
        y=df_patterns['p2_rolling_avg'],
        mode='lines',
        name=f'Jogador 2 (Média móvel {window} games)',
        line=dict(color='#20b2aa', width=2)
    ))
    
    fig_patterns.update_layout(
        title="Tendência de Performance (Média Móvel)",
        xaxis_title="Número do Game",
        yaxis_title="Taxa de Vitória",
        height=chart_height//2
    )
    
    st.plotly_chart(fig_patterns, use_container_width=True)

def show_sample_tennis_data():
    """Mostra dados de exemplo para tênis"""
    st.subheader("🎾 Exemplo de Análise de Tênis")
    st.info("📁 Carregue um arquivo CSV com o histórico de games da partida")
    
    st.markdown("""
    ### 📋 Formato esperado do CSV:
    
    | Coluna | Descrição | Exemplo |
    |--------|-----------|---------|
    | `game_number` | Número sequencial do game | 1, 2, 3... |
    | `player1_score` | Pontos do Jogador 1 no game | 4, 6, 3... |
    | `player2_score` | Pontos do Jogador 2 no game | 2, 4, 6... |
    | `winner` | Vencedor do game (1 ou 2) | 1, 2 |
    | `set_number` | Número do set (opcional) | 1, 2, 3 |
    
    ### 📊 Exemplo de dados:
    ```csv
    game_number,player1_score,player2_score,winner,set_number
    1,4,2,1,1
    2,6,4,1,1
    3,3,6,2,1
    4,4,1,1,1
    5,2,6,2,1
    ```
    """)
    
    # Gerar dados de exemplo
    np.random.seed(42)
    sample_games = []
    
    for i in range(1, 21):  # 20 games de exemplo
        # Simular pontuação realística
        if np.random.random() > 0.5:  # Jogador 1 vence
            p1_score = np.random.choice([4, 5, 6], p=[0.6, 0.2, 0.2])
            p2_score = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.3, 0.2])
            winner = 1
        else:  # Jogador 2 vence
            p2_score = np.random.choice([4, 5, 6], p=[0.6, 0.2, 0.2])
            p1_score = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.3, 0.2])
            winner = 2
        
        set_num = 1 if i <= 12 else 2
        
        sample_games.append({
            'game_number': i,
            'player1_score': p1_score,
            'player2_score': p2_score,
            'winner': winner,
            'set_number': set_num
        })
    
    sample_df = pd.DataFrame(sample_games)
    
    # Mostrar preview dos dados
    st.subheader("👀 Preview dos Dados de Exemplo")
    st.dataframe(sample_df.head(10), use_container_width=True)
    
    # Gráfico de exemplo
    fig_example = go.Figure()
    
    cumsum_p1 = sample_df['winner'].apply(lambda x: 1 if x == 1 else 0).cumsum()
    cumsum_p2 = sample_df['winner'].apply(lambda x: 1 if x == 2 else 0).cumsum()
    
    fig_example.add_trace(go.Scatter(
        x=sample_df['game_number'],
        y=cumsum_p1,
        mode='lines+markers',
        name='Jogador 1',
        line=dict(color='#2e8b57', width=3)
    ))
    
    fig_example.add_trace(go.Scatter(
        x=sample_df['game_number'],
        y=cumsum_p2,
        mode='lines+markers',
        name='Jogador 2',
        line=dict(color='#20b2aa', width=3)
    ))
    
    fig_example.update_layout(
        title="Exemplo: Evolução da Partida",
        xaxis_title="Número do Game",
        yaxis_title="Games Vencidos (Acumulado)",
        height=400
    )
    
    st.plotly_chart(fig_example, use_container_width=True)
    
    st.success("💡 Carregue seu próprio arquivo CSV para análise detalhada da partida!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎾 Dashboard de Análise de Tênis | Desenvolvido com ❤️ usando Streamlit | © 2025</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
