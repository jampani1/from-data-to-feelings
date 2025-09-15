#filtrar e criar uma coluna nova com base em olist_order_reviews_dataset.csv

import numpy as np
import pandas as pd

OR_db = 'csv/olist_order_reviews_dataset.csv'

try:
    # Carregamento dos dados originais
    print("Carregando o dataset de reviews...")
    OR_db = pd.read_csv(OR_db)

    # Filtragem
    print("Filtrando o DataFrame para remover reviews com nota 3...")
    OR_db_filtrado = OR_db[OR_db['review_score'] != 3].copy()
    
    #.copy() para evitar um aviso comum do Pandas (SettingWithCopyWarning)

    # Criação da Variável Alvo ('sentimento')
    print("Criando a nova coluna 'sentimento'...")
    condicao = OR_db_filtrado['review_score'] > 3
    OR_db_filtrado['sentimento'] = np.where(condicao, 'positivo', 'negativo')

    # Verificação do Resultado
    print("\n--- Verificação Concluída ---")
    print("Contagem de reviews por sentimento:")
    print(OR_db_filtrado['sentimento'].value_counts())
    
    print("\nVisualização das 5 primeiras linhas do novo DataFrame:")
    print(OR_db_filtrado.head())

except FileNotFoundError:
    print(f"ERRO: Arquivo não encontrado em '{OR_db}'")