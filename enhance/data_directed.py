import pandas as pd

# 1. Focar nos arquivos que parecem mais relevantes para o nosso objetivo
arquivo_principal = './csv/olist_order_reviews_dataset.csv'

print("--- ANÁLISE DIRECIONADA: Investigando a tabela de Reviews ---")
df_reviews = pd.read_csv(arquivo_principal)

print("\n[INFO] Informações Gerais da Tabela de Reviews:")
df_reviews.info()

print("\n[CONTAGEM] Distribuição das Notas de Avaliação:")
print(df_reviews['review_score'].value_counts(normalize=True).sort_index() * 100)

print("\n[AMOSTRA] Exemplo de Comentários:")
# Filtra para mostrar apenas reviews que de fato têm um comentário escrito
print(df_reviews[df_reviews['review_comment_message'].notna()].head())
# nao aparece no output pois está em uma tabela compactada entre order_id ... review_creation_date

print("\n" + "="*80 + "\n")