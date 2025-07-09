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

def gerar_relatorio_latex(resultados_anova, resultados_tukey, pasta_saida='analise_estatistica'):
    """
    Gera um relatório LaTeX com os resultados das análises estatísticas
    """
    os.makedirs(pasta_saida, exist_ok=True)
    
    with open(os.path.join(pasta_saida, 'relatorio_estatistico.tex'), 'w', encoding='utf-8') as f:
        # Preâmbulo
        f.write(r'\documentclass{article}' + '\n')
        f.write(r'\usepackage[utf8]{inputenc}' + '\n')
        f.write(r'\usepackage{booktabs}' + '\n')
        f.write(r'\usepackage{graphicx}' + '\n')
        f.write(r'\usepackage{float}' + '\n')
        f.write(r'\usepackage{hyperref}' + '\n')
        f.write(r'\title{Relatório de Análise Estatística - Comparação de Modelos}' + '\n')
        f.write(r'\author{Análise Automática}' + '\n')
        f.write(r'\date{\today}' + '\n')
        f.write(r'\begin{document}' + '\n')
        f.write(r'\maketitle' + '\n')
        
        # Introdução
        f.write(r'\section{Introdução}' + '\n')
        f.write(r'Este relatório apresenta a análise estatística comparativa entre diferentes modelos ' + '\n')
        f.write(r'de redes neurais convolucionais para a detecção de fissuras em superfícies de concreto. ' + '\n')
        f.write(r'As análises incluem testes ANOVA e Tukey HSD para verificar diferenças significativas entre os modelos.' + '\n')
        
        # Resultados ANOVA
        f.write(r'\section{Resultados ANOVA}' + '\n')
        f.write(r'A análise de variância (ANOVA) foi utilizada para determinar se há diferenças ' + '\n')
        f.write(r'estatisticamente significativas entre os modelos para cada métrica avaliada.' + '\n')
        
        f.write(r'\begin{table}[H]' + '\n')
        f.write(r'\centering' + '\n')
        f.write(r'\caption{Resultados do Teste ANOVA}' + '\n')
        f.write(r'\begin{tabular}{lccc}' + '\n')
        f.write(r'\toprule' + '\n')
        f.write(r'Métrica & Estatística F & p-valor & Significativo \\' + '\n')
        f.write(r'\midrule' + '\n')
        
        for metrica, resultado in resultados_anova.items():
            significativo = "Sim" if resultado['p-valor'] < 0.05 else "Não"
            f.write(f"{metrica} & {resultado['f_stat']:.4f} & {resultado['p-valor']:.4f} & {significativo} \\\\\n")
        
        f.write(r'\bottomrule' + '\n')
        f.write(r'\end{tabular}' + '\n')
        f.write(r'\end{table}' + '\n')
        
        # Resultados Tukey
        f.write(r'\section{Resultados do Teste Tukey HSD}' + '\n')
        f.write(r'O teste Tukey HSD foi aplicado para as métricas que apresentaram ' + '\n')
        f.write(r'diferenças significativas no teste ANOVA (p < 0.05), permitindo ' + '\n')
        f.write(r'identificar quais pares de modelos diferem entre si.' + '\n')
        
        for metrica, tukey in resultados_tukey.items():
            f.write(f"\n\\subsection{{Teste Tukey para {metrica}}}\n")
            
            # Filtrar apenas resultados significativos
            df_sig = pd.DataFrame(data=tukey._results_table.data[1:], 
                                  columns=tukey._results_table.data[0])
            df_sig = df_sig[df_sig['Significativo'] == 'Sim']
            
            if len(df_sig) > 0:
                f.write(r'\begin{table}[H]' + '\n')
                f.write(r'\centering' + '\n')
                f.write(f"\\caption{{Diferenças significativas para {metrica}}}\n")
                f.write(r'\begin{tabular}{lccc}' + '\n')
                f.write(r'\toprule' + '\n')
                f.write(r'Modelo 1 & Modelo 2 & Diferença Média & p-valor \\' + '\n')
                f.write(r'\midrule' + '\n')
                
                for _, row in df_sig.iterrows():
                    f.write(f"{row['Grupo1']} & {row['Grupo2']} & {row['Diferença Média']:.4f} & {row['p-valor']:.4f} \\\\\n")
                
                f.write(r'\bottomrule' + '\n')
                f.write(r'\end{tabular}' + '\n')
                f.write(r'\end{table}' + '\n')
            else:
                f.write("Não foram encontradas diferenças significativas entre os modelos para esta métrica.\n")
            
            # Incluir boxplot
            f.write(r'\begin{figure}[H]' + '\n')
            f.write(r'\centering' + '\n')
            f.write(f"\\includegraphics[width=0.8\\textwidth]{{boxplot_{metrica}.png}}\n")
            f.write(f"\\caption{{Boxplot comparativo para a métrica {metrica}}}\n")
            f.write(r'\end{figure}' + '\n')
        
        # Conclusão
        f.write(r'\section{Conclusão}' + '\n')
        f.write(r'Com base nos resultados dos testes estatísticos, podemos concluir que: ' + '\n\n')
        
        # Gerar conclusões automaticamente baseadas nos resultados
        metricas_significativas = [m for m, r in resultados_anova.items() if r['p-valor'] < 0.05]
        if metricas_significativas:
            f.write(r'Foram encontradas diferenças estatisticamente significativas entre os modelos para as seguintes métricas: ' + '\n')
            f.write(r'\begin{itemize}' + '\n')
            for m in metricas_significativas:
                f.write(f"\\item {m}\n")
            f.write(r'\end{itemize}' + '\n\n')
        else:
            f.write(r'Não foram encontradas diferenças estatisticamente significativas entre os modelos para nenhuma métrica.' + '\n\n')
        
        f.write(r'Uma análise mais detalhada sobre quais modelos específicos apresentam melhor desempenho ' + '\n')
        f.write(r'pode ser obtida através dos resultados do teste Tukey e dos boxplots apresentados.' + '\n')
        
        f.write(r'\end{document}' + '\n')
    
    print(f"Relatório LaTeX gerado em: {os.path.join(pasta_saida, 'relatorio_estatistico.tex')}")
    return os.path.join(pasta_saida, 'relatorio_estatistico.tex') 