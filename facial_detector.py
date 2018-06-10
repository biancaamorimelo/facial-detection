import cv2

classificador = cv2.CascadeClassifier('cascades\haarcascade_frontalface_default.xml')

imagem = cv2.imread('pessoas/pessoas4.jpg')

imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.1, minNeighbors=9, minSize=(30,30))
# quantidade de faces na imagem
print(len(facesDetectadas))
#X, Y, largura e altura da imagem (da face) - cada linha é uma face
print(facesDetectadas)

for (x, y, l, a) in facesDetectadas:
    #print(x, y, l, a)
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
    #(x,y) = posição original
    #(x + l, y + a) = onde quero desenhar a borda

cv2.imshow("Faces encontradas", imagem)
cv2.waitKey()




