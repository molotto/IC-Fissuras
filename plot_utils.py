from matplotlib.widgets import Button
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

def plot_img_erradas(imagens_erradas, base_dir, model):
    num_imagens = len(imagens_erradas)
    test_dir = base_dir + '/test'
    imagens_por_pagina = 5
    indice_atual = 0
    
    total_paginas = (num_imagens // imagens_por_pagina) + (num_imagens % imagens_por_pagina > 0)

    fig, img_grupo = plt.subplots(1, imagens_por_pagina, figsize=(15, 3))
    plt.subplots_adjust(bottom=0.1)

    def atualizar_imagens():
        nonlocal indice_atual
        inicio_index = indice_atual * imagens_por_pagina
        end_index = min(inicio_index + imagens_por_pagina, num_imagens)

        for img_individual in img_grupo:
            img_individual.clear()
        
        for i in range(inicio_index, end_index):
            try:
                full_img_path = os.path.join(test_dir, imagens_erradas[i])
                img = image.load_img(full_img_path, target_size=(224, 224))
                img_array = image.img_to_array(img) / 255.0
                img_grupo[i - inicio_index].imshow(img_array)
                img_grupo[i - inicio_index].set_title(f'Imagem {i + 1}')
                img_grupo[i - inicio_index].axis('off')
            except FileNotFoundError as e:
                print(f"Erro ao carregar a imagem: {e}")

        plt.draw()

    def proxima_pagina(event):
        nonlocal indice_atual
        if indice_atual < total_paginas - 1:
            indice_atual += 1
            atualizar_imagens()

    def pagina_anterior(event):
        nonlocal indice_atual
        if indice_atual > 0:
            indice_atual -= 1
            atualizar_imagens()

    botao_prox = Button(plt.axes([0.8, 0.05, 0.1, 0.075]), '-->')
    botao_prox.on_clicked(proxima_pagina)

    botao_ant = Button(plt.axes([0.1, 0.05, 0.1, 0.075]), '<--')
    botao_ant.on_clicked(pagina_anterior)

    plot_file = f'plotagem_{model}'
    if not os.path.exists(plot_file):
        os.makedirs(plot_file)

    atualizar_imagens()
    plt.savefig(os.path.join(plot_file, 'erradas.png'))