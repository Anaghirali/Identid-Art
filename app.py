import os
from flask import Flask, request, render_template
import cv2
import numpy as np

app = Flask(__name__)

# Dicionário que mapeia nomes de imagens de referência 
imagens_referencia = {
    'mona.jpg': 'Mona Lisa - Obra famosa de Leonardo da Vinci',
    'vangogh.jpg': 'Descrição de outra obra',
    'lavie.png': 'Descrição de mais uma obra',
}

#método identificador
def identificar_obra_de_arte(imagem_de_entrada):
    resultados = []
    imagem_disponivel = False

    for imagem_referencia_nome, descricao in imagens_referencia.items():
        imagem_referencia = cv2.imread(os.path.join('imgs', imagem_referencia_nome))
        #transformando em branco e preto
        if imagem_referencia is not None:
            imagem_disponivel = True
            imagem_de_entrada_gray = cv2.cvtColor(imagem_de_entrada, cv2.COLOR_BGR2GRAY)
            imagem_referencia_gray = cv2.cvtColor(imagem_referencia, cv2.COLOR_BGR2GRAY)

            sift = cv2.SIFT_create()
            keypoints_entrada, descritores_entrada = sift.detectAndCompute(imagem_de_entrada_gray, None)
            keypoints_referencia, descritores_referencia = sift.detectAndCompute(imagem_referencia_gray, None)

            bf = cv2.BFMatcher()
            correspondencias = bf.knnMatch(descritores_entrada, descritores_referencia, k=2)

            correspondencias_boas = []
            for m, n in correspondencias:
                if m.distance < 0.15 * n.distance:
                    correspondencias_boas.append(m)

            if len(correspondencias_boas) > 10:
                resultados.append(f"A imagem corresponde à obra: {descricao}")

    return resultados

@app.route('/', methods=['GET', 'POST'])
def index():
    resultados = None
    aviso = None

    if request.method == 'POST':
        imagem_de_entrada = request.files['imagem']
        if imagem_de_entrada:
            imagem = cv2.imdecode(np.frombuffer(imagem_de_entrada.read(), np.uint8), cv2.IMREAD_COLOR)
            resultados = identificar_obra_de_arte(imagem)
            if not resultados:
              aviso = "Ainda não conseguimos reconhecer esta obra :( "
            

    return render_template('index.html', resultados=resultados, aviso=aviso)

if __name__ == '__main__':
    app.run(debug=True)
