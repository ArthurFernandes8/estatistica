# Importando as bibliotecas necessárias
import pandas as pd
import numpy as np

# --- CARREGAR E INSPECIONAR OS DADOS (VERSÃO CORRIGIDA) ---
# Lendo o arquivo Excel diretamente
file_path = 'Desempenho_Alunos.xlsx'
try:
    df = pd.read_excel(file_path)
    print("Arquivo Excel 'Desempenho_Alunos.xlsx' carregado com sucesso!\n")
except FileNotFoundError:
    print(f"ERRO: O arquivo '{file_path}' não foi encontrado. Verifique se ele está na mesma pasta que o script 'index.py'.")
    exit()


# 1. Descrição geral da base de dados (Item 2 das Orientações)
print("--- Análise Preliminar dos Dados ---")
num_observacoes, num_variaveis = df.shape
print(f"O conjunto de dados possui {num_observacoes} observações (alunos) e {num_variaveis} variáveis (colunas).\n")

# 2. Tipos de dados das colunas
print("--- Tipos de Dados das Colunas ---")
df.info()

# 3. Verificação de Dados Ausentes (Item 3 das Orientações)
print("\n--- Contagem de Dados Ausentes por Coluna ---")
dados_ausentes = df.isnull().sum()
# Mostra apenas as colunas que de fato têm dados ausentes
print(dados_ausentes[dados_ausentes > 0])

# 4. Análise Descritiva para identificar outliers/erros (Item 3 das Orientações)
print("\n--- Estatísticas Descritivas das Variáveis Numéricas ---")
# O .T transpõe a tabela para facilitar a leitura
print(df.describe().T)

# 5. Identificar especificamente o erro em 'horas_estudo'
print("\n--- Linha(s) com Erro Identificado (horas_estudo < 0) ---")
erro_horas_estudo = df[df['horas_estudo'] < 0]
print(erro_horas_estudo)






# ---------------------------------------------------- #






# --- FASE DE LIMPEZA E CORREÇÃO DOS DADOS ---
print("\n--- Iniciando a Limpeza dos Dados ---")

# 1. Remover a linha com horas_estudo negativa
print(f"Número de linhas antes da remoção do erro: {df.shape[0]}")
df_limpo = df[df['horas_estudo'] >= 0].copy()
print(f"Número de linhas após a remoção do erro: {df_limpo.shape[0]}")

# 2. Preencher (imputar) os valores ausentes com a mediana
# Calculando a mediana para 'horas_estudo' (no dataframe já limpo)
mediana_horas = df_limpo['horas_estudo'].median()
df_limpo['horas_estudo'].fillna(mediana_horas, inplace=True)
print(f"\nValores ausentes em 'horas_estudo' preenchidos com a mediana: {mediana_horas}")

# Calculando a mediana para 'notas_anteriores'
mediana_notas = df_limpo['notas_anteriores'].median()
df_limpo['notas_anteriores'].fillna(mediana_notas, inplace=True)
print(f"Valores ausentes em 'notas_anteriores' preenchidos com a mediana: {mediana_notas}")

# --- VERIFICAÇÃO PÓS-LIMPEZA ---
print("\n--- Verificação de Dados Ausentes Após a Limpeza ---")
print(df_limpo.isnull().sum())

print("\n--- Novas Estatísticas Descritivas (Após Limpeza) ---")
print(df_limpo.describe().T)






# ---------------------------------------------------- #







# --- FASE DE ANÁLISE VISUAL E CORRELAÇÃO ---
import matplotlib.pyplot as plt
import seaborn as sns

print("\n--- Gerando Gráficos da Análise Exploratória ---")

# 1. Heatmap de Correlação (Item 4 das Orientações)
# Seleciona apenas colunas numéricas para a matriz de correlação
df_numerico = df_limpo.select_dtypes(include=np.number)
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(df_numerico.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlação entre Variáveis Numéricas', fontsize=16)
# Salva a imagem na pasta
plt.savefig('matriz_correlacao.png', bbox_inches='tight')
plt.close() # Fecha a figura para não exibir na tela e economizar memória

print("Salvo: 'matriz_correlacao.png'")

# 2. Histogramas para Variáveis Numéricas (Item 3 das Orientações)
df_numerico.hist(bins=20, figsize=(14, 10))
plt.suptitle('Distribuição das Variáveis Numéricas', fontsize=16)
# Salva a imagem na pasta
plt.savefig('histogramas.png', bbox_inches='tight')
plt.close()

print("Salvo: 'histogramas.png'")

# 3. Gráficos de Barra para Variáveis Categóricas (Item 3 das Orientações)
# Cria uma figura com dois subplots (um para cada variável)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Distribuição das Variáveis Categóricas', fontsize=16)

# Gráfico para 'atividade_extracurricular'
sns.countplot(ax=axes[0], x='atividade_extracurricular', data=df_limpo, order=df_limpo['atividade_extracurricular'].value_counts().index)
axes[0].set_title('Atividade Extracurricular')

# Gráfico para 'nivel_socioeconomico'
sns.countplot(ax=axes[1], x='nivel_socioeconomico', data=df_limpo, order=df_limpo['nivel_socioeconomico'].value_counts().index)
axes[1].set_title('Nível Socioeconômico')

# Salva a imagem na pasta
plt.savefig('variaveis_categoricas.png', bbox_inches='tight')
plt.close()

print("Salvo: 'variaveis_categoricas.png'")






# ---------------------------------------------------- #







# --- FASE 3: AJUSTE DO MODELO DE REGRESSÃO INICIAL (VERSÃO CORRIGIDA) ---
import statsmodels.api as sm
import pandas as pd # Adicionado para garantir que pd.get_dummies funcione

print("\n--- Iniciando a Modelagem de Regressão ---")

# 1. Preparar os dados para o modelo
# Convertendo variáveis categóricas para o tipo numérico (int)
df_modelo = pd.get_dummies(df_limpo, columns=['atividade_extracurricular', 'nivel_socioeconomico'], drop_first=True, dtype=int)

# 2. Definir a variável dependente (y) e as independentes (X)
y = df_modelo['desempenho']
X = df_modelo.drop('desempenho', axis=1)

# Adicionando uma linha de verificação para garantir que todos os tipos são numéricos
# print("\nVerificando tipos de dados em X antes de modelar:")
# print(X.dtypes)

# 3. Adicionar uma constante (intercepto) ao modelo
X = sm.add_constant(X)

# 4. Ajustar o modelo de Regressão Linear Múltipla (MRLM)
modelo_inicial = sm.OLS(y, X).fit()

# 5. Imprimir o resumo completo dos resultados do modelo
print("\n--- Resumo do Modelo de Regressão Inicial (Todas as Variáveis) ---")
print(modelo_inicial.summary())






# ---------------------------------------------------- #






# --- FASE 4: SELEÇÃO DE MODELO (BACKWARD ELIMINATION) ---

# --- Passo 1: Remover a variável 'idade' ---
print("\n--- Seleção de Modelo: Removendo 'idade' (maior p-valor) ---")

# Reutilizamos o df_modelo que já tem as dummies
# A única diferença é que vamos remover 'idade' da lista de preditores X
X_passo_1 = X.drop('idade', axis=1)

# Ajustamos um novo modelo
modelo_passo_1 = sm.OLS(y, X_passo_1).fit()

# Imprimimos o resumo do novo modelo
print("\n--- Resumo do Modelo Passo 1 (sem 'idade') ---")
print(modelo_passo_1.summary())

# --- Passo 2: Remover a variável 'nivel_socioeconomico' ---
print("\n--- Seleção de Modelo: Removendo 'nivel_socioeconomico' (maiores p-valores) ---")

# Vamos usar como base o X do passo anterior (X_passo_1)
X_passo_2 = X_passo_1.drop(['nivel_socioeconomico_Médio', 'nivel_socioeconomico_Baixo'], axis=1)

# Ajustamos o terceiro modelo
modelo_passo_2 = sm.OLS(y, X_passo_2).fit()

# Imprimimos o resumo do novo modelo
print("\n--- Resumo do Modelo Passo 2 (sem 'idade' e 'nivel_socioeconomico') ---")
print(modelo_passo_2.summary())





# ---------------------------------------------------- #






# --- FASE 5: ANÁLISE DE DIAGNÓSTICO DO MODELO FINAL (ITEM 7) ---

print("\n--- Gerando Gráficos de Diagnóstico para o Modelo Final ---")

# Extrair os resíduos e os valores previstos do modelo final
residuos = modelo_passo_2.resid
valores_previstos = modelo_passo_2.predict()

# Criar uma figura 2x2 para os gráficos de diagnóstico
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Gráficos de Diagnóstico do Modelo de Regressão', fontsize=16)

# 1. Gráfico de Resíduos vs. Valores Ajustados (Verifica Linearidade e Homocedasticidade)
sns.scatterplot(x=valores_previstos, y=residuos, ax=axes[0, 0])
axes[0, 0].axhline(0, color='red', linestyle='--')
axes[0, 0].set_title('Resíduos vs. Valores Previstos')
axes[0, 0].set_xlabel('Valores Previstos')
axes[0, 0].set_ylabel('Resíduos')

# 2. Gráfico Q-Q dos Resíduos (Verifica Normalidade)
sm.qqplot(residuos, line='s', ax=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q Plot dos Resíduos')

# 3. Gráfico Scale-Location (Verifica Homocedasticidade)
residuos_padronizados_sqrt = np.sqrt(np.abs(modelo_passo_2.get_influence().resid_studentized_internal))
sns.scatterplot(x=valores_previstos, y=residuos_padronizados_sqrt, ax=axes[1, 0])
axes[1, 0].set_title('Scale-Location Plot')
axes[1, 0].set_xlabel('Valores Previstos')
axes[1, 0].set_ylabel('Raiz Quadrada dos Resíduos Padronizados')

# 4. Gráfico de Resíduos vs. Alavancagem (Identifica pontos influentes)
sm.graphics.influence_plot(modelo_passo_2, ax=axes[1, 1], criterion="cooks")
axes[1, 1].set_title('Resíduos vs. Alavancagem')

# Salva a imagem na pasta
plt.savefig('diagnostico_modelo.png', bbox_inches='tight')
plt.close()

print("Salvo: 'diagnostico_modelo.png'")





# ---------------------------------------------------- #





# --- FASE 6: PREVISÕES FINAIS PARA ALUNOS HIPOTÉTICOS (ITEM 8) ---

print("\n--- Gerando Previsões Finais ---")

# 1. Criar um DataFrame com os novos dados dos alunos hipotéticos
# As colunas devem ter EXATAMENTE os mesmos nomes do modelo (X_passo_2.columns)
perfis = pd.DataFrame({
    'const': [1, 1], # A constante (intercepto) é sempre 1
    'horas_estudo': [20, 8],
    'frequencia': [95, 80],
    'motivacao': [22, 12],
    'notas_anteriores': [8.5, 5.0],
    'sono': [7.0, 6.0],
    'atividade_extracurricular_Sim': [1, 0] # 1 para Sim, 0 para Não
})

# 2. Usar o modelo final (modelo_passo_2) para fazer as previsões
previsoes = modelo_passo_2.get_prediction(perfis)
sumario_previsoes = previsoes.summary_frame(alpha=0.05) # alpha=0.05 para 95% de confiança

# 3. Imprimir os resultados de forma organizada
for i in range(len(perfis)):
    aluno = "Aluno A (Dedicado)" if i == 0 else "Aluno B (De Risco)"
    
    previsao_pontual = sumario_previsoes['mean'][i]
    ic_inferior, ic_superior = sumario_previsoes['mean_ci_lower'][i], sumario_previsoes['mean_ci_upper'][i]
    ip_inferior, ip_superior = sumario_previsoes['obs_ci_lower'][i], sumario_previsoes['obs_ci_upper'][i]

    print(f"\n--- Resultados para o {aluno} ---")
    print(f"Previsão Pontual de Desempenho: {previsao_pontual:.2f}")
    print(f"Intervalo de Confiança (95%) para a MÉDIA de desempenho: [{ic_inferior:.2f}, {ic_superior:.2f}]")
    print(f"Intervalo de Predição (95%) para o desempenho INDIVIDUAL: [{ip_inferior:.2f}, {ip_superior:.2f}]")