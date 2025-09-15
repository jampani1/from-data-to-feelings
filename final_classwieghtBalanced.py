
import pandas as pd
import numpy as np
import re
import string

from nltk.corpus import stopwords

# ferramenta para dividir os dados em treino e teste
from sklearn.model_selection import train_test_split
# ferramenta para vetorizar o texto com TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# para o modelo ML
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ferramentas para gerar e visualizar a matriz de confusão
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

OR_db = 'csv/olist_order_reviews_dataset.csv'

try:
    df = pd.read_csv(OR_db)
    df_filtrado = df[df['review_score'] != 3].copy()
    condicao = df_filtrado['review_score'] > 3
    df_filtrado['sentimento'] = np.where(condicao, 'positivo', 'negativo')

    # limpeza de Texto 

    def limpar_texto(texto): #aqui tira numeros, pontuacoes e stopwords (palavras neutras sem peso)
        texto = texto.lower()
        # expressão regular r'\d+' encontra um ou mais dígitos (números)
        texto = re.sub(r'\d+', '', texto)
        # essa linha cria uma "tabela de tradução" que mapeia cada sinal de pontuação para 'nada'
        texto = texto.translate(str.maketrans('', '', string.punctuation))
        texto = texto.strip()
        
        #limpeza pt2
        #  lista de stopwords em português
        palavras_de_parada = set(stopwords.words('portuguese'))
        # separação do texto em uma lista de palavras
        lista_palavras = texto.split()
        # reconstrução  do texto, mantendo apenas as palavras que NÃO estão na lista de stopwords
        texto_sem_stopwords = [palavra for palavra in lista_palavras if palavra not in palavras_de_parada]
        # voltar as palavras juntas
        texto = ' '.join(texto_sem_stopwords)

        return texto

    # preencher qualquer comentário ausente com uma string vazia.
    df_filtrado['review_comment_message'] = df_filtrado['review_comment_message'].fillna('')

    # '.apply()' é um método do Pandas que passa cada item da coluna pela função especificada.
    print("Aplicando a função de limpeza de texto. Isso pode levar alguns segundos...")
    df_filtrado['texto_limpo'] = df_filtrado['review_comment_message'].apply(limpar_texto)

    # visualização do resultado
    # print("\n--- Verificação da Limpeza de Texto ---")
    # .head() para ver as colunas original e limpa, lado a lado.
    # print(df_filtrado[['sentimento', 'review_comment_message', 'texto_limpo']].head())


    # mapeando a variável alvo 'sentimento' para números (y)
    # modelos ML trabalham com números, então 'positivo' -> 1 e 'negativo' -> 0.
    df_filtrado['sentimento_numerico'] = df_filtrado['sentimento'].map({'positivo': 1, 'negativo': 0})

    # definindo nossas variáveis X (features) e y (alvo)
    # X são os dados que o modelo usará para aprender (o texto limpo). - que contem apenas palavras significativas
    # y é o que o modelo tentará prever (o sentimento 0 ou 1).
    X = df_filtrado['texto_limpo']
    y = df_filtrado['sentimento_numerico']

    # divisão dos dados em conjuntos de treino e teste
    # O modelo treina com 80% dos dados e depois validamos sua performance nos 20% que ele nunca viu.
    # random_state=42 garante que a divisão seja sempre a mesma, para resultados consistentes.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # 'stratify=y' garante que a proporção de positivos/negativos seja a mesma no treino e no teste.

    # vetorização do texto com TF-IDF
    # Isso transforma o texto limpo em uma matriz de números que o modelo entende.
    # max_features=5000 foca apenas nas 5000 palavras mais importantes, otimizando a performance.
    vectorizer = TfidfVectorizer(max_features=5000)

    # AQUI APRENDEMOS O VOCABULÁRIO E TRANSFORMAMOS OS DADOS DE TREINO
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # AQUI APENAS TRANSFORMAMOS OS DADOS DE TESTE, USANDO O MESMO VOCABULÁRIO APRENDIDO ANTES
    X_test_tfidf = vectorizer.transform(X_test)

    # verificação
    print("\n--- PREPARAÇÃO FINAL CONCLUÍDA ---")
    print("Formato dos dados de treino (X_train_tfidf):", X_train_tfidf.shape)
    print("Formato dos dados de teste (X_test_tfidf):", X_test_tfidf.shape)
    print("Os dados estão prontos para a modelagem!")

    # modelo ML
    # criando o modelo
    modelo = LogisticRegression(random_state=42, class_weight='balanced')

    # treinando o modelo
    print("\nTreinando o modelo de Regressão Logística...")
    modelo.fit(X_train_tfidf, y_train)
    print("Modelo treinado com sucesso!")

    #  previsões no conjunto de teste
    previsoes = modelo.predict(X_test_tfidf)

    # avaliando a performance
    acuracia = accuracy_score(y_test, previsoes)
    print(f"\nAcurácia do modelo: {acuracia * 100:.2f}%")

    print("\nRelatório de Classificação:")
    # comparação das previsões com os resultados reais (y_test) para gerar o relatório.
    # target_names nos ajuda a rotular as linhas 0 e 1 como 'negativo' e 'positivo'.
    print(classification_report(y_test, previsoes, target_names=['negativo', 'positivo']))

except FileNotFoundError:
    print(f"ERRO: Arquivo não encontrado em '{OR_db}'")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")

print("\n--- ANÁLISE BÔNUS: PRINCIPAIS RAZÕES PARA CADA SENTIMENTO ---")

try:
    # pegando os nomes das palavras (features) que o TF-IDF aprendeu
    # lista das 5000 palavras que o modelo conhece.
    feature_names = vectorizer.get_feature_names_out()

    # pegando os coeficientes (pesos) que o modelo de Regressão Logística aprendeu
    # cada coeficiente corresponde a uma palavra na mesma posição da lista acima.
    coeficientes = modelo.coef_[0]

    # criando um DataFrame para visualizar as palavras e seus pesos de forma organizada
    df_razoes = pd.DataFrame({'palavra': feature_names, 'peso': coeficientes})

    # encontrando as principais razões para reviews POSITIVAS
    # ordenação do DataFrame do maior peso para o menor e pegamos as 15 primeiras.
    razoes_positivas = df_razoes.sort_values(by='peso', ascending=False).head(15)
    
    print("\nPrincipais 15 palavras que indicam uma review POSITIVA:")
    print(razoes_positivas)

    # encontrando as principais razões para reviews NEGATIVAS
    # ordenação do DataFrame do menor peso (mais negativo) para o maior e pegamos as 15 primeiras.
    razoes_negativas = df_razoes.sort_values(by='peso', ascending=True).head(15)
    
    print("\nPrincipais 15 palavras que indicam uma review NEGATIVA:")
    print(razoes_negativas)

except NameError:
    print("\nERRO: Certifique-se de que o 'modelo' e o 'vectorizer' já foram treinados.")
except Exception as e:
    print(f"Ocorreu um erro na análise bônus: {e}")