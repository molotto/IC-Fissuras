import argparse
import numpy as np
import os
from tensorflow.keras.applications import VGG16, VGG19, MobileNet, MobileNetV2, ResNet152V2, DenseNet201
from dataset_utils import carrega_dataset
from model_utils import build_modelo
from train_utils import treino, setup_custom_early_stopping, pesos_classes
from metrics_utils import calculo_metricas, matriz_confusao, grafico_historico, verifica_imagens_erradas
from plot_utils import plot_img_erradas
from seed_utils import setar_semente
from train_utils import executa_cross_validation
from stats_utils import analisar_todas_metricas, carregar_metricas_modelos
setar_semente(654)

def main():
    parser = argparse.ArgumentParser(description='descrição do programa.')
    parser.add_argument('--epocas', type=int, default=10, help='quantidade de épocas')
    parser.add_argument('--model', type=str, default='VGG16', help='Modelo da arquitetura a ser usado')
    args = parser.parse_args()

    epocas = args.epocas
    model = args.model.upper()

    history_file = f'history-{model.lower()}.npy'
    pred_file = f'previsoes-{model.lower()}.npy'
    model_file = f'modelo_{model.lower()}.h5'

    base_dir = "/home/luis/Datasets_Fissuras/dataset_fissuras_hajarzoubir/organized_dataset"
    batch_size = 8
    train_generator, validation_generator, test_generator = carrega_dataset(base_dir, batch_size)

    conv_base = None
    previsoes_classes = None
    identificacoes_teste = test_generator.classes

    print(f"\nModelo escolhido: {model}")

    while True:
        setar_semente(654)
        print("\nEscolha uma opção:")
        print("1 - Treinar o modelo")
        print("2 - Plotar gráfico de histórico de treinamento")
        print("3 - Plotar matriz de confusão")
        print("4 - Plotar imagens erradas") 
        print("5 - Executar tudo")
        print("6 - Executar Cross-Validation")
        print("7 - Executar Cross-Validation para TODOS os modelos")
        print("8 - Realizar testes estatísticos (ANOVA e Tukey)")
        print("9 - Sair")


        escolha = input("Digite o número da sua escolha: ")

        if escolha == '1':
            shape = (224, 224, 3)
            if model == "VGG19":
                 conv_base = VGG19(weights='imagenet', include_top=False, input_shape=shape)
            elif model == "VGG16":
                conv_base = VGG16(weights='imagenet', include_top=False, input_shape=shape)
            elif model == "MOBILENET":
                conv_base = MobileNet(weights='imagenet', include_top=False, input_shape=shape)
            elif model == "MOBILENETV2":
                 conv_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=shape)
            elif model == "RESNET":
                 conv_base = ResNet152V2(weights='imagenet', include_top=False, input_shape=shape)
            elif model == "DENSENET":
                 conv_base = DenseNet201(weights='imagenet', include_top=False, input_shape=shape)
            elif model == "EFFICIENTNET":
                 conv_base = EfficientNetV2S(weights='imagenet', include_top=False, input_shape=shape)
            elif model == "CONVNEXT":
                 conv_base = ConvNeXtSmall(weights='imagenet', include_top=False, input_shape=shape)
            else:
                print(f"Modelo {model} não reconhecido. Use: VGG16, VGG19, MobileNet, MobileNetV2, ResNet152V2, DenseNet201, EfficientNetV2S, ConvNeXtSmall")
                return

            modelo_real = build_modelo(conv_base, shape)

            class_weight = pesos_classes(2772, 1050)
            early_stopping = setup_custom_early_stopping()

            history = treino(modelo_real, train_generator, validation_generator, class_weight, epocas, [early_stopping])

            np.save(history_file, history.history)
            modelo_real.save(model_file)
            previsoes = modelo_real.predict(test_generator, steps=test_generator.samples // test_generator.batch_size + 1, verbose=1)
            previsoes_classes = (previsoes > 0.5).astype(int).flatten()
            np.save(pred_file, previsoes_classes)

            precisao, recall, f1 = calculo_metricas(identificacoes_teste, previsoes_classes)
            print(f"\nPrecisão: {precisao}, Recall: {recall}, F1 Score: {f1}")
            
            print("\nTreinamento concluído.")

        elif escolha == '2':
            try:
                grafico_historico(history_file, model)
                if previsoes_classes is None:
                    try:
                        previsoes_classes = np.load(pred_file)
                    except FileNotFoundError:
                        print(f"Arquivo de previsões {pred_file} não encontrado.")
                        continue
                precisao, recall, f1 = calculo_metricas(identificacoes_teste, previsoes_classes)
                print('\nPrecisão:', precisao)
                print('Recall:', recall)
                print('F1: ', f1)

            except FileNotFoundError:
                print(f"Arquivo de histórico {history_file} não encontrado.")
                
        elif escolha == '3':
            if previsoes_classes is None:
                try:
                    previsoes_classes = np.load(pred_file)
                except FileNotFoundError:
                    print(f"Arquivo de previsões {pred_file} não encontrado.")
                    continue

            verifica_imagens_erradas(previsoes_classes, identificacoes_teste, test_generator)
            matriz_confusao(identificacoes_teste, previsoes_classes, validation_generator.class_indices.keys(), model)

        elif escolha == '4':
            if previsoes_classes is None:
                try:
                    previsoes_classes = np.load(pred_file)
                except FileNotFoundError:
                    print(f"Arquivo de previsões {pred_file} não encontrado.")
                    continue
            arquivos_teste = test_generator.filenames
            imagens_erradas = [arquivos_teste[i] for i, (pred, true_label) in enumerate(zip(previsoes_classes, identificacoes_teste)) if pred != true_label]
            plot_img_erradas(imagens_erradas, base_dir, model)

        elif escolha == '5':
            if previsoes_classes is None:
                try:
                    previsoes_classes = np.load(pred_file)
                except FileNotFoundError:
                    print(f"Arquivo de previsões {pred_file} não encontrado.")
                    continue

            arquivos_teste = test_generator.filenames
            imagens_erradas = [arquivos_teste[i] for i, (pred, true_label) in enumerate(zip(previsoes_classes, identificacoes_teste)) if pred != true_label]

            try:
                grafico_historico(history_file, model)
            except FileNotFoundError:
                print(f"Arquivo de histórico {history_file} não encontrado.")
            matriz_confusao(identificacoes_teste, previsoes_classes, validation_generator.class_indices.keys(), model)
            plot_img_erradas(imagens_erradas, base_dir, model)
            arquivos_teste = test_generator.filenames
            imagens_erradas = [arquivos_teste[i] for i, (pred, true_label) in enumerate(zip(previsoes_classes, identificacoes_teste)) if pred != true_label]

        elif escolha == '6':
            caminho_treino = os.path.join(base_dir, "train")
            executa_cross_validation(model, caminho_treino, k=5, batch_size=batch_size, epochs=epocas)

        elif escolha == '7':
            caminho_treino = os.path.join(base_dir, "train")
            modelos = ['VGG16', 'VGG19', 'MOBILENET', 'MOBILENETV2', 'RESNET', 'DENSENET']
            for nome_modelo in modelos:
                print(f"\nExecutando cross-validation para: {nome_modelo}")
                executa_cross_validation(nome_modelo, caminho_treino, k=5, batch_size=batch_size, epochs=epocas)
                
        elif escolha == '8':
            print("\nIniciando análise estatística (ANOVA e Tukey)")
            try:
                # Carregar todas as métricas
                df_metricas = carregar_metricas_modelos()
                
                # Verificar se há modelos suficientes
                if len(df_metricas['modelo'].unique()) < 2:
                    print("São necessários pelo menos 2 modelos diferentes para realizar os testes estatísticos!")
                    print("   Execute primeiro a validação cruzada para múltiplos modelos (opção 7).")
                    continue
                
                # Realizar análise para todas as métricas
                print("\nRealizando testes estatísticos para todas as métricas...")
                resultados_anova, resultados_tukey = analisar_todas_metricas(df_metricas)
                
                print("\nAnálise estatística concluída com sucesso!")
                print(f"   Resultados salvos em: analise_estatistica/")
                print(f"   - Boxplots das métricas")
                print(f"   - CSV com resultados ANOVA")
                print(f"   - CSV com resultados Tukey (quando aplicável)")
                
            except Exception as e:
                print(f"Erro ao realizar análise estatística: {e}")
        
        elif escolha == '9':
            break
        
        else:
            print("Escolha inválida. Por favor, tente novamente.")

if __name__ == '__main__':
    main()
