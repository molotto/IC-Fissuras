from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def calculo_metricas(identificacoes, previsoes):
    precision = precision_score(identificacoes, previsoes)
    recall = recall_score(identificacoes, previsoes)
    f1 = f1_score(identificacoes, previsoes)    
    return precision, recall, f1

def matriz_confusao(identificacoes, previsoes, class_names, model):
    cm = confusion_matrix(identificacoes, previsoes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predição')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão')

    plot_file = f'plotagem_{model}'
    if not os.path.exists(plot_file):
        os.makedirs(plot_file)

    plt.savefig(os.path.join(plot_file, 'matriz_confusao.png'))

def grafico_historico(history_file, model):
    
    historico = np.load(history_file, allow_pickle=True).item()

    # Gráfico da acurácia do treino e validação
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(historico['accuracy'])
    ax1.plot(historico['val_accuracy'])
    ax1.set_title("Acurácia por épocas")
    ax1.set_xlabel('épocas')
    ax1.set_ylabel('acurácia')
    ax1.legend(['treino', 'validação'])

    # Gráfico das perdas de treino e validação
    ax2.plot(historico['loss'])
    ax2.plot(historico['val_loss'])
    ax2.set_title("Perdas por épocas")
    ax2.set_xlabel('épocas')
    ax2.set_ylabel('perdas')
    ax2.legend(['treino', 'validação'])

    plot_file = f'plotagem_{model}'
    if not os.path.exists(plot_file):
        os.makedirs(plot_file)

    plt.savefig(os.path.join(plot_file, 'grafico.png'))

def verifica_imagens_erradas(previsoes_classes, identificacoes_teste, test_generator):
    
    arquivos_teste = test_generator.filenames
    imagens_erradas = []
    falsos_positivos = []
    falsos_negativos = []

    for i, (pred, true_label) in enumerate(zip(previsoes_classes, identificacoes_teste)):
        if pred != true_label:
            img_path = arquivos_teste[i]
            imagens_erradas.append(img_path)
            if pred == 1 and true_label == 0:
                falsos_positivos.append(img_path)
            elif pred == 0 and true_label == 1:
                falsos_negativos.append(img_path)

    total_imagens = len(identificacoes_teste)
    porcentagem_fp = (len(falsos_positivos) / total_imagens) * 100
    porcentagem_fn = (len(falsos_negativos) / total_imagens) * 100

    print(f'Imagens erradamente classificadas: {len(imagens_erradas)}')
    print(f'Falsos Positivos (rachaduras falsas): {len(falsos_positivos)} - {porcentagem_fp:.2f}%')
    print(f'Falsos Negativos (rachaduras não detectadas): {len(falsos_negativos)} - {porcentagem_fn:.2f}%')
