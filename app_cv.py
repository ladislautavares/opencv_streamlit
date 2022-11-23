import cv2
import streamlit as st
from PIL import Image # para reproduzir a imagem é upada
import numpy as np
from skimage import morphology, color, feature, filters # outros filtros

def brilho_imagem(imagem, resultado):
    img_brilho = cv2.convertScaleAbs(imagem, beta = resultado)
    return img_brilho

def borra_imagem(imagem, resultado):
    img_borrada = cv2.GaussianBlur(imagem, (7, 7), resultado) # (7, 7) é a máscara
    return img_borrada

def melhora_detalhe(imagem):
    img_melhorada = cv2.detailEnhance(imagem, sigma_s = 34, sigma_r = 0.50)
    return img_melhorada
# uma imagem em tons de cinza é melhor para o processamento de imagens
def escala_cinza(imagem):
# BGR = blue, green, red    
    img_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    return img_cinza

def principal():
    st.title('OpenCV Data App')
    st.subheader('Esse aplicativo web permite integrar processamento de imagens com OpenCV')
    st.text('Streamlit OpenCV')
    
# input da imagem    
    arquivo_imagem = st.file_uploader('Envie a sua imagem', type = ['jpg', 'png', 'jpeg'])
    
# cria um controle deslizante na lateral
    taxa_borrao = st.sidebar.slider('Borrão', min_value = 0.2, max_value = 3.5)
# value é o valor padrão    
    qtd_brilho = st.sidebar.slider('Brilho', min_value = -50, max_value = 50, value = 0)
# a pessoa escolhe se quer melhorar a imagem. Se ela for clicada, vai chamar outra função.    
    filtro_aprimoramento = st.sidebar.checkbox('Melhor detalhes da imagem')
# outro checkbox, agora para converter a imagem em tons de cinza    
    img_cinza = st.sidebar.checkbox('Converter para escala de cinza')
    
# os filtros abaixo são da biblioteca skimage    
    img_erosao = st.sidebar.checkbox('Filtro erosão')
    img_dilatacao = st.sidebar.checkbox('Filtro dilatação')
    img_edge = st.sidebar.checkbox('Filtro edge')
    
    
    if not arquivo_imagem: # se o arquivo de imagem não existir, ele não executa o resto do código
        return None
    
    imagem_original = Image.open(arquivo_imagem)
# tem que converter a imagem original para um array numpy, caso contrário o OpenCV não consegue trabalhar

    imagem_original = np.array(imagem_original)
    imagem_processada = borra_imagem(imagem_original, taxa_borrao)
    imagem_processada = brilho_imagem(imagem_processada, qtd_brilho)
    
# verifica se o filtro_aprimoramento (primeiro check box) foi marcado

    if filtro_aprimoramento:
        imagem_processada = melhora_detalhe(imagem_processada)
    
# verifica se o checkbox da tonalidade cinza foi marcado
    if img_cinza:
        imagem_processada = escala_cinza(imagem_processada)
# se o checkbox de img_erosao é marcado, então é chamado o método erosion de morphology        
    if img_erosao:
        imagem_processada = morphology.erosion(imagem_processada)
    if img_dilatacao:
        imagem_processada = morphology.dilation(imagem_processada)
    if img_edge:
        imagem_processada = filters.sobel(imagem_processada)

    
    st.text('Imagem original vs Imagem processada')
    
    st.image([imagem_original, imagem_processada])
    
if __name__ == '__main__':
    principal()