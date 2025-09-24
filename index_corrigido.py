import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# --- CARREGAMENTO DOS DADOS ---
DATA_PATH = 'Desempenho_Alunos.xlsx'
try:
    df = pd.read_excel(DATA_PATH)
    print("Arquivo Excel 'Desempenho_Alunos.xlsx' carregado com sucesso!\n")
except FileNotFoundError:
    print(f"ERRO: O arquivo '{DATA_PATH}' não foi encontrado. Verifique se ele está na mesma pasta que o script.")
    exit()

# --- ANÁLISE PRELIMINAR DOS DADOS ---
print("--- Análise Preliminar dos Dados ---")
num_observacoes, num_variaveis = df.shape
print(f"O conjunto de dados possui {num_observacoes} observações (alunos) e {num_variaveis} variáveis (colunas).\n")

print("--- Tipos de Dados das Colunas ---")
df.info()

print("\n--- Contagem de Dados Ausentes por Coluna ---")
dados_ausentes = df.isnull().sum()
print(dados_ausentes[dados_ausentes > 0])

print("\n--- Estatísticas Descritivas das Variáveis Numéricas ---")
print(df.describe().T)

print("\n--- Linha(s) com Erro Identificado (horas_estudo < 0) ---")
erro_horas_estudo = df[df['horas_estudo'] < 0]
print(erro_horas_estudo)

# --- FASE DE LIMPEZA E CORREÇÃO DOS DADOS ---
print("\n--- Iniciando a Limpeza dos Dados ---")
print(f"Número de linhas antes da remoção do erro: {df.shape[0]}")
df_limpo = df[df['horas_estudo'] >= 0].copy()
print(f"Número de linhas após a remoção do erro: {df_limpo.shape[0]}")

mediana_horas = df_limpo['horas_estudo'].median()
df_limpo['horas_estudo'].fillna(mediana_horas, inplace=True)
print(f"\nValores ausentes em 'horas_estudo' preenchidos com a mediana: {mediana_horas}")

mediana_notas = df_limpo['notas_anteriores'].median()
df_limpo['notas_anteriores'].fillna(mediana_notas, inplace=True)
print(f"Valores ausentes em 'notas_anteriores' preenchidos com a mediana: {mediana_notas}")

print("\n--- Verificação de Dados Ausentes Após a Limpeza ---")
print(df_limpo.isnull().sum())

print("\n--- Novas Estatísticas Descritivas (Após Limpeza) ---")
print(df_limpo.describe().T)

# --- FASE DE ANÁLISE VISUAL E CORRELAÇÃO ---
print("\n--- Gerando Gráficos da Análise Exploratória ---")
df_numerico = df_limpo.select_dtypes(include=np.number)
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(df_numerico.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlação entre Variáveis Numéricas', fontsize=16)
plt.savefig('matriz_correlacao.png', bbox_inches='tight')
plt.close()
print("Salvo: 'matriz_correlacao.png'")

axes = df_numerico.hist(bins=20, figsize=(14, 10))
for ax, col in zip(axes.flatten(), df_numerico.columns):
    ax.set_xlabel(col)
    ax.set_ylabel('Número de Observações')
    ax.set_title(col)
plt.suptitle('Distribuição das Variáveis Numéricas', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('histogramas.png', bbox_inches='tight')
plt.close()
print("Salvo: 'histogramas.png'")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Distribuição das Variáveis Categóricas', fontsize=16)
sns.countplot(ax=axes[0], x='atividade_extracurricular', data=df_limpo, order=df_limpo['atividade_extracurricular'].value_counts().index)
axes[0].set_title('Atividade Extracurricular')
sns.countplot(ax=axes[1], x='nivel_socioeconomico', data=df_limpo, order=df_limpo['nivel_socioeconomico'].value_counts().index)
axes[1].set_title('Nível Socioeconômico')
plt.savefig('variaveis_categoricas.png', bbox_inches='tight')
plt.close()
print("Salvo: 'variaveis_categoricas.png'")

# --- FASE DE MODELAGEM DE REGRESSÃO ---
print("\n--- Iniciando a Modelagem de Regressão ---")
df_modelo = pd.get_dummies(df_limpo, columns=['atividade_extracurricular', 'nivel_socioeconomico'], drop_first=True, dtype=int)
y = df_modelo['desempenho']
X = df_modelo.drop('desempenho', axis=1)
X = sm.add_constant(X)
modelo_inicial = sm.OLS(y, X).fit()
print("\n--- Resumo do Modelo de Regressão Inicial (Todas as Variáveis) ---")
print(modelo_inicial.summary())

# --- FASE DE SELEÇÃO DE MODELO (BACKWARD ELIMINATION) ---
print("\n--- Seleção de Modelo: Removendo 'idade' (maior p-valor) ---")
X_passo_1 = X.drop('idade', axis=1)
modelo_passo_1 = sm.OLS(y, X_passo_1).fit()
print("\n--- Resumo do Modelo Passo 1 (sem 'idade') ---")
print(modelo_passo_1.summary())

print("\n--- Seleção de Modelo: Removendo 'nivel_socioeconomico' (maiores p-valores) ---")
X_passo_2 = X_passo_1.drop(['nivel_socioeconomico_Médio', 'nivel_socioeconomico_Baixo'], axis=1)
modelo_passo_2 = sm.OLS(y, X_passo_2).fit()
print("\n--- Resumo do Modelo Passo 2 (sem 'idade' e 'nivel_socioeconomico') ---")
print(modelo_passo_2.summary())

# --- FASE DE DIAGNÓSTICO DO MODELO FINAL ---
print("\n--- Gerando Gráficos de Diagnóstico para o Modelo Final ---")
residuos = modelo_passo_2.resid
valores_previstos = modelo_passo_2.predict()
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Gráficos de Diagnóstico do Modelo de Regressão', fontsize=16)
sns.scatterplot(x=valores_previstos, y=residuos, ax=axes[0, 0])
axes[0, 0].axhline(0, color='red', linestyle='--')
axes[0, 0].set_title('Resíduos vs. Valores Previstos')
axes[0, 0].set_xlabel('Valores Previstos')
axes[0, 0].set_ylabel('Resíduos')
sm.qqplot(residuos, line='s', ax=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q Plot dos Resíduos')
residuos_padronizados_sqrt = np.sqrt(np.abs(modelo_passo_2.get_influence().resid_studentized_internal))
sns.scatterplot(x=valores_previstos, y=residuos_padronizados_sqrt, ax=axes[1, 0])
axes[1, 0].set_title('Scale-Location Plot')
axes[1, 0].set_xlabel('Valores Previstos')
axes[1, 0].set_ylabel('Raiz Quadrada dos Resíduos Padronizados')
sm.graphics.influence_plot(modelo_passo_2, ax=axes[1, 1], criterion="cooks")
axes[1, 1].set_title('Resíduos vs. Alavancagem')
plt.savefig('diagnostico_modelo.png', bbox_inches='tight')
plt.close()
print("Salvo: 'diagnostico_modelo.png'")

# --- FASE DE PREVISÕES FINAIS PARA ALUNOS HIPOTÉTICOS ---
print("\n--- Gerando Previsões Finais ---")
perfis = pd.DataFrame({
    'const': [1, 1],
    'horas_estudo': [20, 8],
    'frequencia': [95, 80],
    'motivacao': [22, 12],
    'notas_anteriores': [8.5, 5.0],
    'sono': [7.0, 6.0],
    'atividade_extracurricular_Sim': [1, 0]
})
previsoes = modelo_passo_2.get_prediction(perfis)
sumario_previsoes = previsoes.summary_frame(alpha=0.05)
for i in range(len(perfis)):
    aluno = "Aluno A (Dedicado)" if i == 0 else "Aluno B (De Risco)"
    previsao_pontual = sumario_previsoes['mean'][i]
    ic_inferior, ic_superior = sumario_previsoes['mean_ci_lower'][i], sumario_previsoes['mean_ci_upper'][i]
    ip_inferior, ip_superior = sumario_previsoes['obs_ci_lower'][i], sumario_previsoes['obs_ci_upper'][i]
    print(f"\n--- Resultados para o {aluno} ---")
    print(f"Previsão Pontual de Desempenho: {previsao_pontual:.2f}")
    print(f"Intervalo de Confiança (95%) para a MÉDIA de desempenho: [{ic_inferior:.2f}, {ic_superior:.2f}]")
    print(f"Intervalo de Predição (95%) para o desempenho INDIVIDUAL: [{ip_inferior:.2f}, {ip_superior:.2f}]")