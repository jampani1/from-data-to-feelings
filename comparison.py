
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords

# ferramenta para dividir os dados em treino e teste
from sklearn.model_selection import train_test_split
# ferramenta para vetorizar o texto com TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

import unicodedata

# para o modelo ML
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#  gerar e visualizar a matriz de confusão
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# para comparar com outros modelos
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# para script final resposta
import joblib

# -------- CARREGAR E PREPARAR DADOS ----------
def carregar_e_preparar_dados(caminho_arquivo):
    """
    Carrega o dataset, filtra as reviews de nota 3 e cria a coluna de sentimento.
    """
    print("1. Carregando e preparando os dados iniciais...")
    df = pd.read_csv(caminho_arquivo)
    df_filtrado = df[df['review_score'] != 3].copy()
    condicao = df_filtrado['review_score'] > 3
    df_filtrado['sentimento'] = np.where(condicao, 'positivo', 'negativo')
    return df_filtrado

# --------  LIMPEZA DE TEXTO --------------------
def remover_acentos(texto):
    # 'rápida' em 'rapida'
    return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')

def aplicar_limpeza_de_texto(df):
    """
    Aplica a função de limpeza de texto na coluna de comentários.
    """
    print("2. Aplicando a limpeza de texto...")
    
    def limpar_texto(texto):
        texto = str(texto).lower() # Garante que o input é string
        texto = re.sub(r'\d+', '', texto)
        texto = texto.translate(str.maketrans('', '', string.punctuation))
        texto = texto.strip()
        palavras_de_parada = set(stopwords.words('portuguese'))
        lista_palavras = texto.split()
        texto_sem_stopwords = [palavra for palavra in lista_palavras if palavra not in palavras_de_parada]
        texto = ' '.join(texto_sem_stopwords)
        texto = remover_acentos(texto)
        
        return texto

    df['review_comment_message'] = df['review_comment_message'].fillna('')
    df['texto_limpo'] = df['review_comment_message'].apply(limpar_texto)
    return df

# --------  VETORIZACAO E DIVISAO ----------------
def vetorizar_e_dividir_dados(df):
    """
    Converte o texto em vetores TF-IDF e divide os dados em treino e teste.
    """
    print("3. Vetorizando e dividindo os dados...")
    df['sentimento_numerico'] = df['sentimento'].map({'positivo': 1, 'negativo': 0})
    X = df['texto_limpo']
    y = df['sentimento_numerico']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("--- PREPARAÇÃO FINAL CONCLUÍDA ---")
    print(f"Formato dos dados de treino: {X_train_tfidf.shape}")
    print(f"Formato dos dados de teste: {X_test_tfidf.shape}")
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer
# --------------------------------------------------------------------

# --------  TREINAR E AVALIAR MODELOS -----------
def treinar_e_avaliar_modelo(X_train, y_train, X_test, y_test, class_weight=None):
    """
    Treina um modelo de Regressão Logística e exibe seu relatório de performance.
    """
    print(f"\n--- Treinando modelo com class_weight={class_weight or 'Nenhum'} ---")
    modelo = LogisticRegression(random_state=42, class_weight=class_weight)
    modelo.fit(X_train, y_train)
    print("Modelo treinado com sucesso!")
    
    previsoes = modelo.predict(X_test)
    
    acuracia = accuracy_score(y_test, previsoes)
    print(f"Acurácia do modelo: {acuracia * 100:.2f}%")
    print("Relatório de Classificação:")
    print(classification_report(y_test, previsoes, target_names=['negativo', 'positivo']))
    
    return modelo, previsoes # Retornamos para uso posterior

# --------  ANÁLISE  ----------------------
def analisar_razoes_modelo(modelo, vectorizer):
    """
    Extrai e exibe as palavras mais importantes para cada sentimento.
    """
    print("\n--- ANÁLISE: PRINCIPAIS RAZÕES PARA CADA SENTIMENTO ---")
    feature_names = vectorizer.get_feature_names_out()
    coeficientes = modelo.coef_[0]
    df_razoes = pd.DataFrame({'palavra': feature_names, 'peso': coeficientes})
    
    razoes_positivas = df_razoes.sort_values(by='peso', ascending=False).head(15)
    print("\nPrincipais 15 palavras que indicam uma review POSITIVA:")
    print(razoes_positivas)
    
    razoes_negativas = df_razoes.sort_values(by='peso', ascending=True).head(15)
    print("\nPrincipais 15 palavras que indicam uma review NEGATIVA:")
    print(razoes_negativas)
# --------------------------------------------------------------------

# -------- ESSE TRECHO PODERIA ESTAR NO SCRIPT LOGISTIC_REGRESSION.PY ------------------
def comparar_modelos_visualmente(y_test, previsoes_padrao, previsoes_balanceado):
    """
    Cria e salva um gráfico comparando as Matrizes de Confusão de dois modelos.
    """
    print("\n--- Gerando comparação visual do modelo de Regressão Logística com e sem balanceamento ---")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Comparação de Matrizes de Confusão ', fontsize=16)

    # Matriz para o Modelo Padrão
    cm_padrao = confusion_matrix(y_test, previsoes_padrao)
    sns.heatmap(cm_padrao, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
    axes[0].set_title('Modelo Padrão (sem class_weight)')
    axes[0].set_ylabel('Real')
    axes[0].set_xlabel('Previsto')

    # Matriz para o Modelo Balanceado
    cm_balanceado = confusion_matrix(y_test, previsoes_balanceado)
    sns.heatmap(cm_balanceado, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
    axes[1].set_title("Modelo Balanceado (com class_weight)")
    axes[1].set_ylabel('Real')
    axes[1].set_xlabel('Previsto')

    plt.savefig('./img/comparacao_logisticRegression.png')
    print("Gráfico de comparação salvo em './img/comparacao_logisticRegression.png'")

# --------  COMPARAÇÃO COM OUTROS MODELOS ------------------
def comparar_varios_modelos(X_train, y_train, X_test, y_test):
    """
    Treina, avalia e compara múltiplos modelos de classificação.
    Gera uma tabela e um gráfico com os resultados.
    """
    # definição dos modelos que vamos comparar
    modelos = {
        "Regressão Logística": LogisticRegression(random_state=42, class_weight='balanced'),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "SVM": SVC(random_state=42, class_weight='balanced')
    }

    resultados = []
    print("\n--- INICIANDO COMPARAÇÃO DE VÁRIOS MODELOS ---")
    for nome, modelo in modelos.items():
        print(f"\nTreinando o modelo: {nome}...")
        modelo.fit(X_train, y_train)
        previsoes = modelo.predict(X_test)
        
        # Calcular métricas para a classe negativa (pos_label=0)
        precisao = precision_score(y_test, previsoes, pos_label=0)
        recall = recall_score(y_test, previsoes, pos_label=0)
        f1 = f1_score(y_test, previsoes, pos_label=0)
        
        resultados.append({
            "Modelo": nome,
            "Precisão (Negativo)": precisao,
            "Recall (Negativo)": recall,
            "F1-Score (Negativo)": f1
        })
        print(f"Resultados para {nome} calculados.")

    df_resultados = pd.DataFrame(resultados)
    print("\n--- TABELA DE COMPARAÇÃO DE PERFORMANCE (CLASSE NEGATIVA) ---")
    print(df_resultados)

    print("\n--- Gerando gráfico de comparação geral ---")
    df_resultados.set_index('Modelo').plot(kind='bar', figsize=(14, 8))
    plt.title('Comparação de Modelos - Performance na Classe Negativa')
    plt.ylabel('Pontuação da Métrica (0.0 a 1.0)')
    plt.xticks(rotation=25, ha='right') # Melhoramos a rotação para não sobrepor
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig('./img/comparacao_modelos.png')
    print("Gráfico salvo em './img/comparacao_modelos.png'")

# -------- EXECUÇÃO PRINCIPAL ------------------
if __name__ == "__main__":
    caminho_arquivo = 'csv/olist_order_reviews_dataset.csv'
    
    try:
        # Etapa 1: Carregar os dados
        df_inicial = carregar_e_preparar_dados(caminho_arquivo)
        
        # Etapa 2: Limpar o texto
        df_limpo = aplicar_limpeza_de_texto(df_inicial)
        
        # Etapa 3: Vetorizar e dividir
        X_train, X_test, y_train, y_test, vectorizer = vetorizar_e_dividir_dados(df_limpo)
        
        # Etapa 4: Treinar e avaliar os dois modelos
        modelo_padrao, previsoes_padrao = treinar_e_avaliar_modelo(X_train, y_train, X_test, y_test)
        modelo_balanceado, previsoes_balanceado = treinar_e_avaliar_modelo(X_train, y_train, X_test, y_test, class_weight='balanced')
        
        # Etapa 5: Analisar o modelo final (balanceado)
        analisar_razoes_modelo(modelo_balanceado, vectorizer)

        # Etapa 6: Gerar a comparação visual
        comparar_modelos_visualmente(y_test, previsoes_padrao, previsoes_balanceado)

        # Etapa 7: Comparar com outros modelos
        comparar_varios_modelos(X_train, y_train, X_test, y_test)

        # Etapa 8: Salvar os artefatos do melhor modelo para uso futuro
        # Escolhemos o modelo_balanceado e o vectorizer que foram treinados com todos os dados.
        print("\n--- Salvando o modelo final e o vetorizador ---")

        joblib.dump(modelo_balanceado, './to_predict/modelo_sentimento.joblib')
        joblib.dump(vectorizer, './to_predict/vetorizador_tfidf.joblib')

        print("Modelo e vetorizador salvos como 'modelo_sentimento.joblib' e 'vetorizador_tfidf.joblib' em ./to_predict/")

    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado em '{caminho_arquivo}'")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")