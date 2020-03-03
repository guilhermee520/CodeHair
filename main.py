from sklearn.externals import joblib
import cv2
from image_related.image_constructor import get_mask, apply_mask
from image_related.image_predictor import extractDominantColor, plotColorBar
from matplotlib import pyplot as plt


model = joblib.load('models/classificador_cor_cabelo.weights')

#image = cv2.imread("test/test_images/cabelo_claro/20_0_1_20170113132705262.jpg")
image = cv2.imread("test/test_images/cabelo_escuro/Frame00570-org.jpg")
mask = get_mask(image)
builded_image = apply_mask(image, mask)
colorInformation = extractDominantColor(builded_image, hasThresholding=True)
dominantColors = colorInformation[0].get('color') + colorInformation[1].get('color') + colorInformation[2].get('color') + colorInformation[3].get('color') + colorInformation[4].get('color')

colour_bar = plotColorBar(colorInformation)
plt.axis("off")
plt.imshow(colour_bar)
plt.title("Color Bar")
plt.show()