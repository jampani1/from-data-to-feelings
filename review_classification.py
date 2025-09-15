# carrega o modelo treinado e classifica todas as reviews de um arquivo.

import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
import joblib

# Função de Limpeza de Texto
# (mesma função do script de treino para garantir consistência)
def aplicar_limpeza_de_texto(df):
    """
    Aplica a função de limpeza de texto na coluna de comentários.
    """
    print("Aplicando a limpeza de texto...")
    
    def limpar_texto(texto):
        texto = str(texto).lower()
        texto = re.sub(r'\d+', '', texto)
        # (Opcional: adicione a remoção de acentos aqui se desejar)
        texto = texto.translate(str.maketrans('', '', string.punctuation))
        texto = texto.strip()
        palavras_de_parada = set(stopwords.words('portuguese'))
        lista_palavras = texto.split()
        texto_sem_stopwords = [palavra for palavra in lista_palavras if palavra not in palavras_de_parada]
        texto = ' '.join(texto_sem_stopwords)
        return texto

    df['review_comment_message'] = df['review_comment_message'].fillna('')
    df['texto_limpo'] = df['review_comment_message'].apply(limpar_texto)
    return df

# --- Bloco de Execução Principal ---
if __name__ == "__main__":
    try:
        # Carrega modelo e vetorizador salvos
        print("Carregando modelo e vetorizador pré-treinados...")
        modelo = joblib.load('./to_predict/modelo_sentimento.joblib')
        vectorizer = joblib.load('./to_predict/vetorizador_tfidf.joblib')

        # Carrega o dataset original completo que vai ser classificado
        caminho_arquivo_original = 'csv/olist_order_reviews_dataset.csv'
        df_original = pd.read_csv(caminho_arquivo_original)
        
        # Aplica a limpeza de texto
        df_classificar = aplicar_limpeza_de_texto(df_original.copy())
        
        # Vetoriza os textos usando o vetorizador carregado
        print("Vetorizando textos...")
        textos_vetorizados = vectorizer.transform(df_classificar['texto_limpo'])

        # Faz as previsões usando o modelo carregado
        print("Classificando reviews...")
        previsoes_finais = modelo.predict(textos_vetorizados)
        
        # Prepara a tabela de resultado
        mapa_sentimento = {0: 'negativo', 1: 'positivo'}
        df_classificar['sentimento_previsto'] = np.vectorize(mapa_sentimento.get)(previsoes_finais)
        
        df_resultado_final = df_classificar[[
            'review_id', 'order_id', 'review_score', 
            'review_comment_message', 'sentimento_previsto'
        ]]
        
        # Salva o resultado final em um arquivo CSV
        nome_arquivo_saida = 'final_reviews.csv'
        df_resultado_final.to_csv(nome_arquivo_saida, index=False)
        
        print(f"\nProcesso concluído! Tabela final salva como '{nome_arquivo_saida}'")
        print("\nVisualização do resultado:")
        print(df_resultado_final.head())

    except FileNotFoundError:
        print("ERRO: Certifique-se que os arquivos 'modelo_sentimento.joblib',")
        print("'vetorizador_tfidf.joblib' e 'olist_order_reviews_dataset.csv' estão na mesma pasta.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")