import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
import warnings

def carregar_metricas_modelos(pasta_metricas='metricas_cross_val'):
    """
    Carrega todas as métricas de diferentes modelos em um único DataFrame
    """
    todos_dfs = []
    for arquivo in os.listdir(pasta_metricas):
        if arquivo.startswith('metricas_') and arquivo.endswith('.csv'):
            caminho = os.path.join(pasta_metricas, arquivo)
            df = pd.read_csv(caminho)
            todos_dfs.append(df)
    
    if not todos_dfs:
        raise ValueError(f"Nenhum arquivo de métricas encontrado em {pasta_metricas}")
    
    return pd.concat(todos_dfs, ignore_index=True)

def realizar_anova(df_metricas, metrica='acuracia'):
    """
    Realiza teste ANOVA para verificar se há diferença significativa entre modelos
    para uma métrica específica.
    """
    modelos = df_metricas['modelo'].unique()
    
    # Criar grupos por modelo
    grupos = [df_metricas[df_metricas['modelo'] == modelo][metrica].values for modelo in modelos]
    
    # Realizar ANOVA
    f_stat, p_valor = stats.f_oneway(*grupos)
    
    print(f"\n== Teste ANOVA para {metrica.upper()} ==")
    print(f"Estatística F: {f_stat:.4f}")
    print(f"Valor p: {p_valor:.4f}")
    
    if p_valor < 0.05:
        print(f"RESULTADO: Há diferença estatisticamente significativa entre os modelos para {metrica} (p < 0.05)")
    else:
        print(f"RESULTADO: Não há diferença estatisticamente significativa entre os modelos para {metrica} (p >= 0.05)")
    
    return f_stat, p_valor

def realizar_tukey(df_metricas, metrica='acuracia', alpha=0.05):
    """
    Realiza teste de Tukey HSD para comparações múltiplas par a par entre modelos
    para uma métrica específica.
    """
    # Configurar MultiComparison
    mc = MultiComparison(df_metricas[metrica], df_metricas['modelo'])
    
    # Realizar teste de Tukey
    result = mc.tukeyhsd(alpha=alpha)
    
    print(f"\n== Teste de Tukey HSD para {metrica.upper()} (alpha={alpha}) ==")
    print(result)
    
    # Resumir resultados em forma mais legível
    resultados = []
    
    # Acessar os resultados de forma segura
    try:
        for i in range(len(result.pvalues)):
            if hasattr(result, 'data') and isinstance(result.data, np.ndarray) and result.data.ndim >= 2:
                grupo1 = result.groupsunique[result.data[i,0]]
                grupo2 = result.groupsunique[result.data[i,1]]
            else:
                # Abordagem alternativa quando os dados não estão no formato esperado
                pares = list(zip(mc.groupsunique[mc.pairindices[0]], mc.groupsunique[mc.pairindices[1]]))
                if i < len(pares):
                    grupo1, grupo2 = pares[i]
                else:
                    grupo1, grupo2 = f"Grupo_{i*2}", f"Grupo_{i*2+1}"
                    
            diferenca = result.meandiffs[i]
            p_valor = result.pvalues[i]
            significativo = p_valor < alpha
            resultados.append({
                'Grupo1': grupo1,
                'Grupo2': grupo2,
                'Diferença Média': diferenca,
                'p-valor': p_valor,
                'Significativo': 'Sim' if significativo else 'Não'
            })
    except Exception as e:
        print(f"Aviso: Problema ao processar resultados do Tukey: {e}")
        # Extrair informações diretamente do objeto result
        print("Tentando método alternativo para extrair resultados...")
        
        # Obter os nomes dos grupos únicos
        grupos_unicos = mc.groupsunique
        
        # Criar todas as combinações possíveis de pares
        import itertools
        pares = list(itertools.combinations(grupos_unicos, 2))
        
        # Tentar extrair as diferenças médias e p-valores
        if hasattr(result, 'meandiffs') and hasattr(result, 'pvalues'):
            diffs = result.meandiffs
            pvals = result.pvalues
            
            # Verificar se o número de pares corresponde aos resultados
            if len(pares) == len(diffs) == len(pvals):
                for i, (g1, g2) in enumerate(pares):
                    resultados.append({
                        'Grupo1': g1,
                        'Grupo2': g2,
                        'Diferença Média': diffs[i],
                        'p-valor': pvals[i],
                        'Significativo': 'Sim' if pvals[i] < alpha else 'Não'
                    })
            else:
                print("Erro: O número de pares não corresponde ao número de diferenças/p-valores.")
        else:
            print("Erro: Não foi possível extrair diferenças médias e p-valores.")
    
    df_resultados = pd.DataFrame(resultados)
    return result, df_resultados

def plotar_boxplot_modelos(df_metricas, metrica='acuracia'):
    """
    Cria um boxplot comparando os modelos para uma métrica específica
    """
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='modelo', y=metrica, data=df_metricas)
    plt.title(f'Comparação de {metrica.upper()} entre Modelos')
    plt.xlabel('Modelo')
    plt.ylabel(metrica.capitalize())
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    # Salvar gráfico
    pasta_saida = 'analise_estatistica'
    os.makedirs(pasta_saida, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_saida, f'boxplot_{metrica}.png'))
    plt.close()

def analisar_todas_metricas(df_metricas=None, pasta_metricas='metricas_cross_val', metricas=None):
    """
    Realiza análise completa (ANOVA + Tukey) para todas as métricas especificadas
    """
    if df_metricas is None:
        df_metricas = carregar_metricas_modelos(pasta_metricas)
        
    if metricas is None:
        metricas = ['acuracia', 'precisao', 'recall', 'f1_score', 'mcc', 'tempo_inferencia_ms']
    
    # Criar pasta para resultados
    pasta_saida = 'analise_estatistica'
    os.makedirs(pasta_saida, exist_ok=True)
    
    resultados_anova = {}
    resultados_tukey = {}
    
    # Realizar testes para cada métrica
    for metrica in metricas:
        criar_boxplot(df_metricas, metrica, pasta_saida)
    
    # Realizar ANOVA e Tukey
    for metrica in metricas:
        # ANOVA
        grupos = [group[metrica].values for name, group in df_metricas.groupby('modelo')]
        f_stat, p_valor = stats.f_oneway(*grupos)
        resultados_anova[metrica] = {'f_stat': f_stat, 'p-valor': p_valor}
        
        # Tukey (apenas se ANOVA for significativo)
        if p_valor < 0.05:
            tukey = pairwise_tukeyhsd(df_metricas[metrica], df_metricas['modelo'])
            resultados_tukey[metrica] = tukey
    
    # Salvar resultados em CSV
    df_anova = pd.DataFrame(resultados_anova).T
    df_anova.to_csv(os.path.join(pasta_saida, 'resultados_anova.csv'))
    
    for metrica, tukey in resultados_tukey.items():
        pd.DataFrame(data=tukey._results_table.data[1:], 
                    columns=tukey._results_table.data[0]).to_csv(
                    os.path.join(pasta_saida, f'tukey_{metrica}.csv'), index=False)
    
    print(f"\nAnálise concluída. Resultados salvos em '{pasta_saida}'")
    return resultados_anova, resultados_tukey

    return os.path.join(pasta_saida, 'relatorio_estatistico.tex') 
