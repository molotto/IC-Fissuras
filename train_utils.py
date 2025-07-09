from tensorflow.keras.callbacks import Callback
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, VGG19, MobileNet, MobileNetV2, ResNet152V2, DenseNet201
from model_utils import build_modelo
from seed_utils import setar_semente
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

def pesos_classes(qtd_background, qtd_cracks):
    return {0: 1.0, 1: (qtd_background / qtd_cracks)}

class CustomEarlyStopping(Callback):
    def __init__(self, patience=5, min_delta=0.001, verbose=1):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.wait = 0
        self.best_weights = None
        self.best_epoch = 0
        self.best_accuracy = 0
        self.best_val_accuracy = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current_accuracy = logs.get('accuracy')
        current_val_accuracy = logs.get('val_accuracy')

        if np.greater(current_accuracy, current_val_accuracy) and (
            np.greater(current_val_accuracy - self.min_delta, self.best_val_accuracy)
        ):
            self.best_epoch = epoch
            self.best_accuracy = current_accuracy
            self.best_val_accuracy = current_val_accuracy
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.verbose:
                    print(f"\nEpoch {self.stopped_epoch + 1}: early stopping")
                if self.best_weights is not None:
                    print(f"Restaurando os pesos da melhor época ({self.best_epoch + 1}) com accuracy = {self.best_accuracy} e val_accuracy = {self.best_val_accuracy}")
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f"Treinamento interrompido na época {self.stopped_epoch + 1}. Melhor época restaurada: {self.best_epoch + 1}")

def setup_custom_early_stopping():
    return CustomEarlyStopping(patience=5, min_delta=0.001, verbose=1)

def treino(model, train_generator, validation_generator, class_weight, epochs, callbacks):
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    return history

def executa_cross_validation(model_name, train_dir, k=5, batch_size=8, epochs=10):
    print(f"\n Iniciando Cross-Validation para o modelo: {model_name}")

    setar_semente(654)

    dados = []
    for classe in ['Background', 'Cracks']:
        classe_dir = os.path.join(train_dir, classe)
        for nome_img in os.listdir(classe_dir):
            dados.append({
                'caminho': os.path.join(classe_dir, nome_img),
                'classe': classe  # ← mantém como string!
            })

    df = pd.DataFrame(dados)
    df = df.sample(frac=1, random_state=654).reset_index(drop=True)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=654)

    datagen = ImageDataGenerator(rescale=1./255)
    metricas_acuracia = []
    metricas_precisao = []
    metricas_recall = []
    metricas_f1 = []
    metricas_mcc = []
    tempos_inferencia = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df['caminho'], df['classe'])):
        print(f"\n Fold {fold+1}/{k}")

        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]

        train_gen = datagen.flow_from_dataframe(
            df_train,
            x_col='caminho',
            y_col='classe',
            target_size=(224, 224),
            class_mode='binary',
            batch_size=batch_size,
            shuffle=True
        )

        val_gen = datagen.flow_from_dataframe(
            df_val,
            x_col='caminho',
            y_col='classe',
            target_size=(224, 224),
            class_mode='binary',
            batch_size=batch_size,
            shuffle=False
        )

        input_shape = (224, 224, 3)

        if model_name.upper() == 'VGG16':
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        elif model_name.upper() == 'VGG19':
            base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
        elif model_name.upper() == 'MOBILENET':
            base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
        elif model_name.upper() == 'MOBILENETV2':
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        elif model_name.upper() == 'RESNET':
            base_model = ResNet152V2(weights='imagenet', include_top=False, input_shape=input_shape)
        elif model_name.upper() == 'DENSENET':
            base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=input_shape)
        else:
            raise ValueError(f"Modelo '{model_name}' não suportado.")

        model = build_modelo(base_model, input_shape)
        early_stopping = setup_custom_early_stopping()

        qtd_bg = len(df_train[df_train['classe'] == 'Background'])
        qtd_ck = len(df_train[df_train['classe'] == 'Cracks'])
        class_weight = pesos_classes(qtd_bg, qtd_ck)

        history = treino(model, train_gen, val_gen, class_weight, epochs, [early_stopping])
        
        # Avaliação com métricas expandidas
        import time
        start_time = time.time()
        preds_prob = model.predict(val_gen, verbose=0)
        end_time = time.time()
        
        tempo_inferencia = (end_time - start_time) / len(val_gen)
        tempos_inferencia.append(tempo_inferencia)
        
        preds_classes = (preds_prob > 0.5).astype(int).flatten()
        true_classes = val_gen.classes
        
        acc = accuracy_score(true_classes, preds_classes)
        prec = precision_score(true_classes, preds_classes, zero_division=0)
        rec = recall_score(true_classes, preds_classes, zero_division=0)
        f1 = f1_score(true_classes, preds_classes, zero_division=0)
        mcc = matthews_corrcoef(true_classes, preds_classes)
        
        print(f" Fold {fold+1} - Acurácia: {acc:.4f}, Precisão: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}")
        print(f" Tempo médio de inferência: {tempo_inferencia*1000:.2f} ms por batch")
        
        metricas_acuracia.append(acc)
        metricas_precisao.append(prec)
        metricas_recall.append(rec)
        metricas_f1.append(f1)
        metricas_mcc.append(mcc)

    print(f"\nMédias finais:")
    print(f"   Acc: {np.mean(metricas_acuracia):.4f}±{np.std(metricas_acuracia):.4f}")
    print(f"   Prec: {np.mean(metricas_precisao):.4f}±{np.std(metricas_precisao):.4f}")
    print(f"   Rec: {np.mean(metricas_recall):.4f}±{np.std(metricas_recall):.4f}")
    print(f"   F1: {np.mean(metricas_f1):.4f}±{np.std(metricas_f1):.4f}")
    print(f"   MCC: {np.mean(metricas_mcc):.4f}±{np.std(metricas_mcc):.4f}")
    print(f"   Tempo: {np.mean(tempos_inferencia)*1000:.2f}±{np.std(tempos_inferencia)*1000:.2f}ms/batch")

    df_metricas = pd.DataFrame({
        'fold': list(range(1, k + 1)),
        'modelo': [model_name.upper()] * k,
        'acuracia': metricas_acuracia,
        'precisao': metricas_precisao,
        'recall': metricas_recall,
        'f1_score': metricas_f1,
        'mcc': metricas_mcc,
        'tempo_inferencia_ms': [t*1000 for t in tempos_inferencia],
        'tipo': ['crossval'] * k
    })

    os.makedirs('metricas_cross_val', exist_ok=True)
    caminho_csv = os.path.join('metricas_cross_val', f'metricas_{model_name.upper()}.csv')
    df_metricas.to_csv(caminho_csv, index=False)

    print(f"\n Métricas salvas em: {caminho_csv}")
    return df_metricas
