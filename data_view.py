import pandas as pd
import os

csv_path = './csv/'

try:
    all_arq = os.listdir(csv_path)
    arquivos_csv = [arq for arq in all_arq if arq.endswith('.csv')] # Lista apenas arquivos .csv
    if not arquivos_csv:
        print("Nenhum arquivo .csv encontrado no diretório.")
    else:
        print(f"Encontrados {len(arquivos_csv)} arquivos .csv. Lendo cada um deles...")
        for nome_arquivo in arquivos_csv:
            print(f"--- Visualizando as 10 primeiras linhas de: {nome_arquivo} ---\n")
            caminho_completo = os.path.join(csv_path, nome_arquivo)
            
            df = pd.read_csv(caminho_completo)
            print(df.head(10))
            
            print("\n" + "="*80 + "\n")

except Exception as e:
    print(f"Ocorreu um erro: {e}")