import math
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2
from tensorflow import keras
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pycm import ConfusionMatrix
from sklearn.metrics import auc, roc_curve
import os
from itertools import cycle
from scipy import interp
from tensorflow.keras.utils import to_categorical
import matplotlib
import math
from IPython.display import clear_output

map_normal_anormal = {
    'normal_superficiel': 0,
    'normal_intermediate': 0,
    'normal_columnar': 0,
    'light_dysplastic': 1,
    'moderate_dysplastic': 1,
    'severe_dysplastic': 1,
    'carcinoma_in_situ': 1
}

def get_size_statistics(source):
    heights = []
    widths = []
    folders = os.listdir(source)

    for folder in folders:
        directorio = os.path.join(source, folder)
        for f in os.listdir(directorio):
            archivo = os.path.join(directorio, f)
            data = np.array(Image.open(archivo)) #PIL Image library
            heights.append(data.shape[0])
            widths.append(data.shape[1])
            
    avg_height = sum(heights) / len(heights)
    avg_width = sum(widths) / len(widths)

    print("Average Height: " + str(avg_height))
    print("Max Height: " + str(max(heights)))
    print("Min Height: " + str(min(heights)))
    print('\n')
    print("Average Width: " + str(avg_width))
    print("Max Width: " + str(max(widths)))
    print("Min Width: " + str(min(widths)))


def stitch_images(images, margin=5, cols=5):
    """Utility function to stitch images together with a `margin`.
    Args:
        images: The array of 2D images to stitch.
        margin: The black border margin size between images (Default value = 5)
        cols: Max number of image cols. New row is created when number of images exceed the column size.
            (Default value = 5)
    Returns:
        A single numpy image array comprising of input images.
    """
    if len(images) == 0:
        return None

    h, w, c = images[0].shape
    n_rows = int(math.ceil(len(images) / cols))
    n_cols = min(len(images), cols)

    out_w = n_cols * w + (n_cols - 1) * margin
    out_h = n_rows * h + (n_rows - 1) * margin
    stitched_images = np.zeros((out_h, out_w, c), dtype=images[0].dtype)

    for row in range(n_rows):
        for col in range(n_cols):
            img_idx = row * cols + col
            if img_idx >= len(images):
                break

            stitched_images[(h + margin) * row: (h + margin) * row + h,
                            (w + margin) * col: (w + margin) * col + w, :] = images[img_idx]

    return stitched_images

def draw_text(img, text, position=(10, 10), font='/home/marco/ConvCervix/fonts/computer-modern/cmunssdc.ttf', font_size=14, color=(0, 0, 0)):
    """Draws text over the image. Requires PIL.
    Args:
        img: The image to use.
        text: The text string to overlay.
        position: The text (x, y) position. (Default value = (10, 10))
        font: The ttf or open type font to use. (Default value = 'FreeSans.ttf')
        font_size: The text font size. (Default value = 12)
        color: The (r, g, b) values for text color. (Default value = (0, 0, 0))
    Returns: Image overlayed with text.
    """

    font = ImageFont.truetype(font, font_size)
    
    # Don't mutate original image
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    draw.fontmode = "0"
    draw.text(position, text, fill=color, font=font)
    return np.asarray(img)

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def sub_image(image, center, theta, width, height, dx=10, dy=10): 
    if 45 < theta <= 90:
        theta = theta - 90
        width, height = height, width
    x = center[0] + 0.5
    y = center[1] + 0.5
    v_x = (math.cos(theta), math.sin(theta)) 
    v_y = (-math.sin(theta), math.cos(theta))
    s_x = x - v_x[0] * (width / 2) - v_y[0] * (height / 2) + dx 
    s_y = y - v_x[1] * (width / 2) - v_y[1] * (height / 2) + dy
    mapping = np.array([[v_x[0],v_y[0], s_x], [v_x[1],v_y[1], s_y]])
    return cv2.warpAffine(image, mapping, (width, height), flags=cv2.WARP_INVERSE_MAP+cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def centro(image):
    ret,thresh = cv2.threshold(image[:,:,0],128,255,0)
    M = cv2.moments(thresh)
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return cX, cY

def imagenes_muestreo(imagenes, figsize=(8, 40), nom_archivo='muestras'):
    plt.figure(figsize=figsize)
    plt.imshow(imagenes)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(f"{nom_archivo}.pdf", bbox_inches='tight', pad_inches=0, dpi=500, facecolor='w')
    plt.savefig(f"{nom_archivo}.png", bbox_inches='tight', pad_inches=0, dpi=500, facecolor='w')
    plt.show()
    
def translate_metric(x):
    translations = {'accuracy': "Exactitud", 'loss': "Perdida"}
    if x in translations:
        return translations[x]
    else:
        return x


class PlotLosses(keras.callbacks.Callback):
    def __init__(self, figsize=None):
        super(PlotLosses, self).__init__()
        self.tamano_fig = figsize

    def on_train_begin(self, logs={}):

        self.base_metrics = [metric for metric in self.params['metrics'] if not metric.startswith('val_')]
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        # clear_output(wait=True)
        self.logs.append(logs.copy())

        plt.figure(figsize=self.tamano_fig)
        
        for metric_id, metric in enumerate(self.base_metrics):
            plt.subplot(1, len(self.base_metrics), metric_id + 1)
            
            plt.plot(range(1, len(self.logs) + 1),
                     [log[metric] for log in self.logs],
                     label="training")
            plt.plot(range(1, len(self.logs) + 1),
                     [log['val_' + metric] for log in self.logs],
                     label="validation")
            plt.title(translate_metric(metric))
            plt.xlabel('Epocas')
            plt.legend(loc='center left')
        plt.tight_layout()
        plt.show()
        
        
def grafica_kfold(cv, X, y, n_splits, num_classes, nom_archivo, dpi=500, lw=20):
    fig, ax = plt.subplots(figsize=(15,4))
    cmap_data = plt.cm.tab10
    cmap_cv = plt.cm.Blues
    """Create a sample plot for indices of a cross-validation object."""

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X, y)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)
    # Plot the data classes and groups at the end
    scat = ax.scatter(range(len(X)), [ii + 1.5] * len(X), 
               c=y.map(map_normal_anormal), 
               marker='_', 
               lw=lw, 
               cmap=cmap_data)
    
    handles = scat.legend_elements()[0]
    labels = y.unique()

    legend1 = ax.legend(handles, 
                        labels,
                        loc="lower center", 
                        title="Clases",  ncol=num_classes)
    
    ax.add_artist(legend1)
    yticklabels = list(range(n_splits)) + ['Clase']
    ax.set(yticks=np.arange(n_splits+2) + .5, 
           yticklabels=yticklabels, 
           xlabel='Índice de la muestra', 
           ylabel="Iteración de validación cruzada (K)", 
           ylim=[n_splits+2.2, -.2], 
           xlim=[0, len(X)])
    ax.set_title(f'{type(cv).__name__} para {num_classes} clases', fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{nom_archivo}.pdf', dpi=dpi, bbox_inches='tight', facecolor='w')
    plt.savefig(f'{nom_archivo}.png', dpi=dpi, bbox_inches='tight', facecolor='w')

def plot_confusion_matrix(cm, archivo,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues
                          ):
    """
    This function modified to plots the ConfusionMatrix object.
    Normalization can be applied by setting `normalize=True`.
    
    Code Reference : 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    """
    plt.figure(figsize=(10,10))
    plt_cm = []
    for i in cm.classes :
        row=[]
        for j in cm.classes:
            row.append(cm.table[i][j])
        plt_cm.append(row)
    plt_cm = np.array(plt_cm)
    accuracy = np.trace(plt_cm) / float(np.sum(plt_cm))
    misclass = 1 - accuracy
    if normalize:
        plt_cm = plt_cm.astype('float') / plt_cm.sum(axis=1)[:, np.newaxis]
        title = f'{title} (normalizada)'
        archivo = f'{archivo}_norm'
    plt.imshow(plt_cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(cm.classes))
    plt.xticks(tick_marks, cm.classes, rotation=45, ha='right')
    plt.yticks(tick_marks, cm.classes)

    fmt = '.2f' if normalize else 'd'
    thresh = plt_cm.max() / 2.
    for i, j in itertools.product(range(plt_cm.shape[0]), range(plt_cm.shape[1])):
        plt.text(j, i, format(plt_cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if plt_cm[i, j] > thresh else "black")
    
    # plt.tight_layout()
    plt.ylabel('Clase actual')
    plt.xlabel('Clase predicción\nExactitud={:0.4f}; Error={:0.4f}'.format(accuracy, misclass))
    plt.savefig(f'{archivo}.png', dpi=500, bbox_inches='tight', facecolor='w')
    plt.savefig(f'{archivo}.pdf', dpi=500, bbox_inches='tight', facecolor='w')
    plt.show()
    
def exp_smooth(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def grafica_metricas(epocas, metrica, metrica_val, etiqueta, titulo, archivo, suavizado=False):
    if suavizado:
        metrica = exp_smooth(metrica)
        metrica_val = exp_smooth(metrica_val)
    plt.figure(figsize=(10,8))
    plt.plot(epocas, metrica, 'b--', label=f'{etiqueta} en entrenamiento')
    plt.plot(epocas, metrica_val, 'r', label=f'{etiqueta} en validación')
    plt.title(titulo)
    plt.xlabel('Épocas')
    plt.ylabel(f'{etiqueta}')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'{archivo}.png', dpi=500, bbox_inches='tight', facecolor='w')
    plt.savefig(f'{archivo}.pdf', dpi=500, bbox_inches='tight', facecolor='w')
    plt.show()

def variabilidad_metricas(df_metricas, etiqueta, titulo, archivo, fontsize=8):
    plt.figure(figsize=(10,8))

    # Create the boxplot
    # bp = plt.boxplot(df_metricas)

    bp = plt.boxplot(df_metricas, patch_artist=True)

    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set( color='#7570b3', linewidth=2)
        # change fill color
        box.set( facecolor = '#1b9e77' )

    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)

    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)

    plt.title(titulo)
    # plt.xlabel('Épocas')
    plt.ylabel(etiqueta)
    plt.gca().xaxis.set_major_locator(matplotlib.ticker.AutoLocator())
    # plt.xticks(np.arange(0,len(df_metricas.index),10))
    plt.xticks([])
    plt.savefig(f'{archivo}.png', dpi=500, facecolor='w', bbox_inches='tight')
    plt.savefig(f'{archivo}.pdf', dpi=500, facecolor='w', bbox_inches='tight')
    plt.show()
    
def boxplot_metricas(df_metricas, titulo, colores, archivo):
    plt.figure(figsize=(10,8))
    df_boxloss = pd.melt(df_metricas, var_name="Métrica", value_name="Valor")
    ax = sns.boxenplot("Métrica", y="Valor", data=df_boxloss, palette=colores)
    ax.set_title(titulo)
    plt.tight_layout()
    plt.savefig(f'{archivo}.png', dpi=500, bbox_inches='tight', facecolor='w')
    plt.savefig(f'{archivo}.pdf', dpi=500, bbox_inches='tight', facecolor='w')
    
def evaluar_clasificacion(datos, num_classes, folder):
    '''
    reporte_completo = Todo
    reporte_clasificacion = metricas clasicas de clasificacion
    reporte_metricas_malas = metricas que queremos que sean minimas
    reporte_metricas_diagnostico = metricas especificas para problemas de diagnostico medico
    reporte_metricas_clasificacion = metricas basicas
    reporte_cero = reporte de metricas que tienen que tender a cero
    reporte_uno = reporte de metricas que tienen que tender a uno
    reporte_html = reporte completo en html
    '''
    cm = ConfusionMatrix(actual_vector=datos[f'Class_cat_{num_classes}'].values, 
                         predict_vector=datos[f'Class_cat_{num_classes}_pred'].values)
    res_conf = ['TPR', 'TNR', 'PPV', 'NPV', 'FNR', 'FPR', 'FDR', 'FOR', 'ACC', 
                'F1', 'BM', 'PRE', 'J', 'CEN', 'MCEN', 'AUC']
    res_conf_mal = ['ERR', 'FNR', 'FPR', 'FDR', 'FOR']
    res_conf_diag = ['PLR', 'NLR', 'DOR', 'DP', 'IS']
    res_class = ['TP', 'TN', 'FP', 'FN', 'P', 'N', 'POP']
    buenas_cero = ['FNR', 'FPR', 'FDR', 'FOR',  'CEN', 'MCEN', 'ERR']
    buenas_uno = ['TPR', 'TNR', 'PPV', 'NPV','ACC', 'F1', 'BM','J', 'AUC']
    cm.save_csv(f'{folder}reporte_completo')
    cm.save_csv(f'{folder}reporte_clasificacion', class_param=res_conf)
    cm.save_csv(f'{folder}reporte_metricas_malas', class_param=res_conf_mal)
    cm.save_csv(f'{folder}reporte_metricas_diagnostico', class_param=res_conf_diag)
    cm.save_csv(f'{folder}reporte_metricas_clasificacion', class_param=res_class)
    cm.save_csv(f'{folder}reporte_cero', class_param=buenas_cero)
    cm.save_csv(f'{folder}reporte_uno', class_param=buenas_uno)
    cm.save_html(f'{folder}reporte_html')
    return cm

def mostrar_reporte(reporte):
    df_reporte = pd.read_csv(reporte)
    df_reporte.rename(columns={'Class': "Métricas" }, inplace=True)
    df_reporte = df_reporte.set_index('Métricas')
    display(df_reporte)
    
def grafica_reporte_uno(num_classes, archivo):
    df_conf_uno = pd.read_csv(f'reporte_{num_classes}_class/reporte_uno.csv')
    df_conf_uno.rename(columns={'Class': "Métricas" }, inplace=True)
    df_conf_uno = df_conf_uno.set_index('Métricas')
    plt.figure(figsize=(10,10))

    ax = sns.heatmap(df_conf_uno, cmap='RdBu', annot=False, square=True, vmin=0.9940, vmax=1, robust=True)
    ax.set_title(r'Reporte de clasificación de {} clases $m \approx 1$'.format(num_classes))
    ax.set_xlabel('Clases')
    for item in ax.get_xticklabels():
        item.set_rotation(30)
        item.set_ha('right')
    for item in ax.get_yticklabels():
        item.set_rotation(0)
    plt.savefig(f'{archivo}.png', dpi=500, bbox_inches='tight', facecolor='w')
    plt.savefig(f'{archivo}.pdf', dpi=500, bbox_inches='tight', facecolor='w')
    plt.show()
    
def grafica_reporte_cero(num_classes, archivo):
    df_conf_cero = pd.read_csv(f'reporte_{num_classes}_class/reporte_cero.csv')
    df_conf_cero.rename(columns={'Class': "Métricas" }, inplace=True)
    df_conf_cero = df_conf_cero.set_index('Métricas')
    plt.figure(figsize=(10,10))
    ax = sns.heatmap(df_conf_cero, cmap='RdBu_r', annot=False, square=True, robust=True)
    ax.set_title(r'Reporte de clasificación de {} clases $m \approx 0$'.format(num_classes))
    ax.set_xlabel('Clases')
    for item in ax.get_xticklabels():
        item.set_rotation(30)
        item.set_ha('right')
    for item in ax.get_yticklabels():
        item.set_rotation(0)
    plt.savefig(f'{archivo}.png', dpi=500, bbox_inches='tight', facecolor='w')
    plt.savefig(f'{archivo}.pdf', dpi=500, bbox_inches='tight', facecolor='w')
    plt.show()
    
def grafica_roc(datos, num_classes, archivo):
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(datos['y'].values, 
                                                       datos['y_pred'].values)
    auc_keras = auc(fpr_keras, tpr_keras)
    plt.figure(figsize=(10,8))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label=f'Área bajo la curva (AUC) = {auc_keras}')
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de falsos negativos')
    plt.title('Curva ROC')
    plt.legend(loc='best')
    plt.savefig(f'{archivo}.png', dpi=500, bbox_inches='tight', facecolor='w')
    plt.savefig(f'{archivo}.pdf', dpi=500, bbox_inches='tight', facecolor='w')
    plt.show()

    
def roc_multiclass(n_classes, y, ypred, class_list, archivo):
    y = to_categorical(y)
    ypred = to_categorical(ypred)
    # Plot linewidth.
    lw = 2
    n_classes = n_classes
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:,i], ypred[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y[:,i], ypred[:,i])
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Zoom in view of the upper left corner.
    plt.figure(figsize=(10,8))
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot(fpr["micro"], tpr["micro"],
             label=f"Micro-promedio de curva ROC: {roc_auc['micro']:.4f}",
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label=f"Macro-promedio de curva ROC: {roc_auc['macro']:.4f}",
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'forestgreen', 'orchid', 'darkblue', 'olive'])
    for i, color in zip(range(len(class_list)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label=f'AUC {class_list[i]} = {roc_auc[i]:.4f}')

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de falsos negativos')
    plt.title('Curva ROC multiclase')
    plt.legend(loc='lower right', prop={'size': 10})
    plt.savefig(f'{archivo}.png', dpi=500, bbox_inches='tight', facecolor='w')
    plt.savefig(f'{archivo}.pdf', dpi=500, bbox_inches='tight', facecolor='w')
    plt.show()
    
def imagen_muestras_erroneas(df_muestras, num_classes, folder,font='./fonts/computer-modern/cmunssdc.ttf', size = 27):
    ims = []
    for archivo, y, yy, in zip(df_muestras['file'], 
                               df_muestras['Class_cat_7'], 
                               df_muestras[f'Class_cat_{num_classes}_pred']):
        img = (cv2.imread(archivo))
        img = draw_text(img, f'y: {y}', color=(0, 255, 0), font_size=size,position=(10, 0), font=font) 
        img = draw_text(img, f'y_pred: {yy}', color=(255, 0, 0), font_size=size,position=(10, 225), font=font)
        ims.append(img)

    stitched = stitch_images(ims, cols=num_classes)    
    plt.figure(figsize=(8, 40))
    plt.axis('off')
    plt.imshow(stitched)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(f'{folder}.png', dpi=500,bbox_inches='tight', pad_inches=0, facecolor='w')
    plt.savefig(f'{folder}.pdf', dpi=500,bbox_inches='tight', pad_inches=0, facecolor='w')   
    plt.show()
    

def heatmap(file, x, y, **kwargs):
    plt.figure(figsize=(10,8))
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = [1]*len(x)

    if 'palette' in kwargs:
        palette = kwargs['palette']
        n_colors = len(palette)
    else:
        n_colors = 256 # Use 256 colors for the diverging color palette
        palette = sns.color_palette("Blues", n_colors) 

    if 'color_range' in kwargs:
        color_min, color_max = kwargs['color_range']
    else:
        color_min, color_max = min(color), max(color) # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    def value_to_color(val):
        if color_min == color_max:
            return palette[-1]
        else:
            val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            ind = int(val_position * (n_colors - 1)) # target index in the color palette
            return palette[ind]

    if 'size' in kwargs:
        size = kwargs['size']
    else:
        size = [1]*len(x)

    if 'size_range' in kwargs:
        size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]
    else:
        size_min, size_max = min(size), max(size)

    size_scale = kwargs.get('size_scale', 500)

    def value_to_size(val):
        if size_min == size_max:
            return 1 * size_scale
        else:
            val_position = (val - size_min) * 0.99 / (size_max - size_min) + 0.01 # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            return val_position * size_scale
    if 'x_order' in kwargs: 
        x_names = [t for t in kwargs['x_order']]
    else:
        x_names = [t for t in sorted(set([v for v in x]))]
    x_to_num = {p[1]:p[0] for p in enumerate(x_names)}

    if 'y_order' in kwargs: 
        y_names = [t for t in kwargs['y_order']]
    else:
        y_names = [t for t in sorted(set([v for v in y]))]
    y_to_num = {p[1]:p[0] for p in enumerate(y_names)}

    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1) # Setup a 1x10 grid
    ax = plt.subplot(plot_grid[:,:-1]) # Use the left 14/15ths of the grid for the main plot

    marker = kwargs.get('marker', 's')

    kwargs_pass_on = {k:v for k,v in kwargs.items() if k not in [
         'color', 'palette', 'color_range', 'size', 'size_range', 'size_scale', 'marker', 'x_order', 'y_order'
    ]}

    ax.scatter(
        x=[x_to_num[v] for v in x],
        y=[y_to_num[v] for v in y],
        marker=marker,
        s=[value_to_size(v) for v in size], 
        c=[value_to_color(v) for v in color],
        **kwargs_pass_on
    )
    ax.set_xticks([v for k,v in x_to_num.items()])
    ax.set_xticklabels([k for k in x_to_num], rotation=45, horizontalalignment='right')
    ax.set_yticks([v for k,v in y_to_num.items()])
    ax.set_yticklabels([k for k in y_to_num])

    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_facecolor('#F1F1F1')
    plt.title('Correlación de características')
    # Add color legend on the right side of the plot
    if color_min < color_max:
        ax = plt.subplot(plot_grid[:,-1]) # Use the rightmost column of the plot

        col_x = [0]*len(palette) # Fixed x coordinate for the bars
        bar_y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars

        bar_height = bar_y[1] - bar_y[0]
        ax.barh(
            y=bar_y,
            width=[5]*len(palette), # Make bars 5 units wide
            left=col_x, # Make bars start at 0
            height=bar_height,
            color=palette,
            linewidth=0
        )
        ax.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
        ax.grid(False) # Hide grid
        ax.set_facecolor('white') # Make background white
        ax.set_xticks([]) # Remove horizontal ticks
        ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3)) # Show vertical ticks for min, middle and max
        ax.yaxis.tick_right() # Show vertical ticks on the right 
    plt.savefig(f'{file}.png', dpi=500, bbox_inches='tight', facecolor='w')
    plt.savefig(f'{file}.pdf', dpi=500, bbox_inches='tight', facecolor='w')
    plt.show()

def corrplot(data, file, size_scale=500, marker='s'):
    corr = pd.melt(data.reset_index(), id_vars='index')
    corr.columns = ['x', 'y', 'value']
    heatmap(file, 
        corr['x'], corr['y'],
        color=corr['value'], color_range=[-1, 1],
        palette=sns.diverging_palette(20, 220, n=256),
        size=corr['value'].abs(), size_range=[0,1],
        marker=marker,
        x_order=data.columns,
        y_order=data.columns[::-1],
        size_scale=size_scale
    )

    
