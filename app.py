import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
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
st.markdown("### Digite seus dados históricos e faça predições!")
st.markdown("---")

# Inicializar dados no session state
if 'match_data' not in st.session_state:
    st.session_state.match_data = []

# Função para treinar modelo com dados do usuário
@st.cache_data
def train_model_with_data(matches_data):
    """Treina modelo com dados fornecidos pelo usuário"""
    if len(matches_data) < 5:
        return None, None, None, None
    
    try:
        df = pd.DataFrame(matches_data)
        
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
            j1_wins = len(df[df['vencedor'] == row['jogador1']])
            j1_total = len(df[(df['jogador1'] == row['jogador1']) | (df['jogador2'] == row['jogador1'])])
            j1_winrate = j1_wins / j1_total if j1_total > 0 else 0.5
            
            j2_wins = len(df[df['vencedor'] == row['jogador2']])
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
        X = np.array(features)
        y = np.array(targets)
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        return model, le_player, le_surface, all_players
        
    except Exception as e:
        st.error(f"Erro ao treinar modelo: {str(e)}")
        return None, None, None, None

# Layout principal
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("✍️ Adicionar Nova Partida")
    
    with st.form("add_match_form"):
        # Inputs para nova partida
        jogador1 = st.text_input("Jogador 1", placeholder="Ex: Novak Djokovic")
        jogador2 = st.text_input("Jogador 2", placeholder="Ex: Rafael Nadal")
        
        col_winner, col_surface = st.columns(2)
        with col_winner:
            vencedor = st.selectbox("Vencedor", ["", jogador1 if jogador1 else "Jogador 1", jogador2 if jogador2 else "Jogador 2"])
        with col_surface:
            superficie = st.selectbox("Superfície", ["Hard", "Clay", "Grass", "Indoor"])
        
        col_date, col_tournament = st.columns(2)
        with col_date:
            data_partida = st.date_input("Data", value=date.today())
        with col_tournament:
            torneio = st.text_input("Torneio (opcional)", placeholder="Ex: ATP Masters")
        
        submitted = st.form_submit_button("➕ Adicionar Partida", type="primary")
        
        if submitted:
            if jogador1 and jogador2 and vencedor and jogador1 != jogador2:
                nova_partida = {
                    'jogador1': jogador1,
                    'jogador2': jogador2,
                    'vencedor': vencedor,
                    'superficie': superficie,
                    'data': data_partida,
                    'torneio': torneio if torneio else "N/A"
                }
                st.session_state.match_data.append(nova_partida)
                st.success(f"✅ Partida adicionada: {vencedor} venceu!")
                st.rerun()
            else:
                st.error("Por favor, preencha todos os campos obrigatórios corretamente")

    # Mostrar dados atuais
    if st.session_state.match_data:
        st.subheader(f"📊 Dados Atuais ({len(st.session_state.match_data)} partidas)")
        
        # Opções de gerenciamento
        col_clear, col_download = st.columns(2)
        with col_clear:
            if st.button("🗑️ Limpar Todos", type="secondary"):
                st.session_state.match_data = []
                st.rerun()
        
        # Mostrar últimas partidas
        df_display = pd.DataFrame(st.session_state.match_data)
        df_display = df_display.sort_values('data', ascending=False).head(10)
        st.dataframe(df_display, use_container_width=True)
        
        # Estatísticas rápidas
        all_players = list(set([p['jogador1'] for p in st.session_state.match_data] + 
                              [p['jogador2'] for p in st.session_state.match_data]))
        surfaces = list(set([p['superficie'] for p in st.session_state.match_data]))
        
        col_stats1, col_stats2 = st.columns(2)
        with col_stats1:
            st.metric("👥 Jogadores", len(all_players))
        with col_stats2:
            st.metric("🏟️ Superfícies", len(surfaces))

with col2:
    st.subheader("🔮 Fazer Predição")
    
    if len(st.session_state.match_data) >= 5:
        # Treinar modelo
        model, le_player, le_surface, all_players = train_model_with_data(st.session_state.match_data)
        
        if model is not None:
            st.success("🤖 Modelo treinado com sucesso!")
            
            # Interface de predição
            with st.form("prediction_form"):
                pred_jogador1 = st.selectbox("Selecione Jogador 1", sorted(all_players))
                pred_jogador2 = st.selectbox("Selecione Jogador 2", 
                                           [p for p in sorted(all_players) if p != pred_jogador1])
                pred_superficie = st.selectbox("Superfície da Partida", 
                                             sorted(list(set([p['superficie'] for p in st.session_state.match_data]))))
                
                predict_button = st.form_submit_button("🎯 Predizer Resultado", type="primary")
                
                if predict_button:
                    try:
                        df = pd.DataFrame(st.session_state.match_data)
                        
                        # Calcular estatísticas para predição
                        j1_wins = len(df[df['vencedor'] == pred_jogador1])
                        j1_total = len(df[(df['jogador1'] == pred_jogador1) | (df['jogador2'] == pred_jogador1)])
                        j1_winrate = j1_wins / j1_total if j1_total > 0 else 0.5
                        
                        j2_wins = len(df[df['vencedor'] == pred_jogador2])
                        j2_total = len(df[(df['jogador1'] == pred_jogador2) | (df['jogador2'] == pred_jogador2)])
                        j2_winrate = j2_wins / j2_total if j2_total > 0 else 0.5
                        
                        # Confronto direto
                        confrontos = df[((df['jogador1'] == pred_jogador1) & (df['jogador2'] == pred_jogador2)) |
                                       ((df['jogador1'] == pred_jogador2) & (df['jogador2'] == pred_jogador1))]
                        j1_wins_h2h = len(confrontos[confrontos['vencedor'] == pred_jogador1])
                        total_h2h = len(confrontos)
                        h2h_advantage = j1_wins_h2h / total_h2h if total_h2h > 0 else 0.5
                        
                        # Preparar features
                        feature_row = [
                            le_player.transform([pred_jogador1])[0],
                            le_player.transform([pred_jogador2])[0],
                            j1_winrate,
                            j2_winrate,
                            le_surface.transform([pred_superficie])[0],
                            h2h_advantage,
                            total_h2h
                        ]
                        
                        # Fazer predição
                        prob = model.predict_proba([feature_row])[0]
                        prediction = model.predict([feature_row])[0]
                        
                        # Mostrar resultados
                        winner = pred_jogador1 if prediction == 1 else pred_jogador2
                        confidence = max(prob) * 100
                        
                        st.success(f"🏆 **Vencedor Previsto: {winner}**")
                        st.info(f"📊 **Confiança: {confidence:.1f}%**")
                        
                        # Gráfico de probabilidades
                        fig = go.Figure(data=[
                            go.Bar(x=[pred_jogador1, pred_jogador2], 
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
                        
                        # Estatísticas detalhadas
                        st.subheader("📈 Análise Detalhada")
                        
                        col_j1, col_j2 = st.columns(2)
                        with col_j1:
                            st.markdown(f"**{pred_jogador1}**")
                            st.write(f"📊 Taxa de Vitória: {j1_winrate:.1%}")
                            st.write(f"🎮 Total de Jogos: {j1_total}")
                        
                        with col_j2:
                            st.markdown(f"**{pred_jogador2}**")
                            st.write(f"📊 Taxa de Vitória: {j2_winrate:.1%}")
                            st.write(f"🎮 Total de Jogos: {j2_total}")
                        
                        if total_h2h > 0:
                            st.markdown("**🤝 Confronto Direto:**")
                            st.write(f"{pred_jogador1}: {j1_wins_h2h} vitórias")
                            st.write(f"{pred_jogador2}: {total_h2h - j1_wins_h2h} vitórias")
                            st.write(f"Total de confrontos: {total_h2h}")
                        else:
                            st.info("Nenhum confronto direto registrado")
                        
                    except Exception as e:
                        st.error(f"Erro na predição: {str(e)}")
        else:
            st.error("Erro ao treinar o modelo")
    else:
        st.info(f"📝 Adicione pelo menos 5 partidas para fazer predições (atual: {len(st.session_state.match_data)})")

# Seção de análise (só aparece se há dados suficientes)
if len(st.session_state.match_data) >= 3:
    st.markdown("---")
    st.subheader("📊 Análise dos Dados")
    
    df_analysis = pd.DataFrame(st.session_state.match_data)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Vitórias por superfície
        surface_counts = df_analysis['superficie'].value_counts()
        fig1 = px.pie(values=surface_counts.values, names=surface_counts.index,
                      title="Distribuição por Superfície",
                      color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col4:
        # Top jogadores por vitórias
        winner_counts = df_analysis['vencedor'].value_counts().head(8)
        fig2 = px.bar(x=winner_counts.values, y=winner_counts.index, orientation='h',
                      title="Jogadores com Mais Vitórias",
                      color=winner_counts.values,
                      color_continuous_scale='viridis')
        fig2.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig2, use_container_width=True)

# Seção de entrada rápida
st.markdown("---")
st.subheader("⚡ Entrada Rápida de Múltiplas Partidas")

with st.expander("📝 Adicionar várias partidas de uma vez"):
    st.markdown("Digite as partidas no formato: `Jogador1 vs Jogador2 | Vencedor | Superfície | Data`")
    
    bulk_input = st.text_area(
        "Partidas (uma por linha):",
        placeholder="""Novak Djokovic vs Rafael Nadal | Novak Djokovic | Hard | 2023-01-15
Carlos Alcaraz vs Jannik Sinner | Carlos Alcaraz | Clay | 2023-05-20
Roger Federer vs Andy Murray | Roger Federer | Grass | 2023-07-10""",
        height=150
    )
    
    if st.button("➕ Adicionar Todas as Partidas"):
        if bulk_input.strip():
            lines = bulk_input.strip().split('\n')
            added_count = 0
            errors = []
            
            for i, line in enumerate(lines, 1):
                try:
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 4:
                        vs_part = parts[0].split(' vs ')
                        if len(vs_part) == 2:
                            nova_partida = {
                                'jogador1': vs_part[0].strip(),
                                'jogador2': vs_part[1].strip(),
                                'vencedor': parts[1],
                                'superficie': parts[2],
                                'data': datetime.strptime(parts[3], '%Y-%m-%d').date(),
                                'torneio': parts[4] if len(parts) > 4 else "N/A"
                            }
                            st.session_state.match_data.append(nova_partida)
                            added_count += 1
                        else:
                            errors.append(f"Linha {i}: Formato incorreto do confronto")
                    else:
                        errors.append(f"Linha {i}: Dados insuficientes")
                except Exception as e:
                    errors.append(f"Linha {i}: {str(e)}")
            
            if added_count > 0:
                st.success(f"✅ {added_count} partidas adicionadas com sucesso!")
                if errors:
                    st.warning(f"⚠️ {len(errors)} erro(s) encontrado(s):")
                    for error in errors:
                        st.write(f"• {error}")
                st.rerun()
            else:
                st.error("Nenhuma partida foi adicionada. Verifique o formato.")

# Footer com instruções
st.markdown("---")
st.markdown("""
<div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0;'>
    <h4>📋 Como Usar:</h4>
    <ol>
        <li><b>Adicione partidas</b> uma por uma usando o formulário acima</li>
        <li><b>Use entrada rápida</b> para adicionar várias de uma vez</li>
        <li><b>Mínimo 5 partidas</b> necessárias para fazer predições</li>
        <li><b>Selecione jogadores</b> e faça sua predição</li>
        <li><b>Analise resultados</b> com gráficos e estatísticas</li>
    </ol>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; color: #666; margin-top: 20px;'>
    <p>🎾 <strong>Tennis Match Predictor</strong> - Digite seus dados e faça predições!</p>
</div>
""", unsafe_allow_html=True)
