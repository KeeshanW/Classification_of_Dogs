# Load all imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import pickle
import seaborn as sns

y_test = np.load('y_test.npy')
y_pred_VGG16 = np.load('y_pred_VVG16.npy')
y_pred_VGG19 = np.load('y_pred_VVG19.npy')

with open("Img_class_set.txt", "rb") as fp:   # Unpickling
  Img_class_set = pickle.load(fp)

cf_matrix_VGG16 = confusion_matrix(y_pred_VGG16, y_test)
cf_matrix_VGG19 = confusion_matrix(y_pred_VGG19, y_test)

accuracy = accuracy_score(y_pred_VGG16, y_test)
# precision = precision_score(y_pred, y_test)
# reacall = recall_score(y_pred, y_test)
# f1 = f1_score(y_pred, y_test)
group_counts = ["{0:0.0f}\n".format(value) for value in cf_matrix_VGG16 .flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cf_matrix_VGG16 .flatten()/np.sum(cf_matrix_VGG16 )]
box_labels = [f"{v1}{v2}".strip() for v1, v2 in zip(group_counts,group_percentages)]
box_labels = np.asarray(box_labels).reshape(cf_matrix_VGG16 .shape[0],cf_matrix_VGG16 .shape[1])

plt.figure(figsize = (24, 16))
plt.rcParams['font.size'] = 14
sns.heatmap(cf_matrix_VGG16 , annot=box_labels, fmt='', cmap='GnBu',xticklabels=Img_class_set,yticklabels=Img_class_set)
plt.yticks(rotation = 0, va = 'center')
plt.xticks(rotation = 45, ha = 'center')
plt.xlabel('True Label', FontSize = 20)
plt.ylabel('Predicted Label', FontSize = 20)
plt.title('VGG-16 Model - Accuracy: {:.2%} ({} Test Images)'.format(accuracy, len(y_test)))
# plt.savefig('/content/drive/MyDrive/Colab Notebooks/Classifictaion_of_Dogs/CF_mat_VVG16.jpg')
# plt.savefig('CF_mat_VVG16-R2.jpg')