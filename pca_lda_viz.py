import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from io import StringIO

from bottle import route, run, request, static_file
import csv
from matplotlib.font_manager import FontProperties
import colorsys

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.lda import LDA

html = '''
<html>
    <body>
        <img src="data:image/png;base64,{}" />
    </body>
</html>
'''

@route('/')
def root():
    return static_file('upload.html', root='.')

@route('/plot', method='POST')
def plot():

   # Get the data
   upload = request.files.get('upload')
   mydata = list(csv.reader(upload.file, delimiter=','))

   x = [row[0:-1] for row in mydata[1:len(mydata)]]

   classes =  [row[len(row)-1] for row in mydata[1:len(mydata)]]
   labels = list(set(classes))
   labels.sort()

   classIndices = np.array([labels.index(myclass) for myclass in classes])

   X = np.array(x).astype('float')
   y = classIndices
   target_names = labels

   #Apply dimensionality reduction
   pca = PCA(n_components=2)
   X_r = pca.fit(X).transform(X)

   lda = LDA(n_components=2)
   X_r2 = lda.fit(X, y).transform(X)

    #Create 2D visualizations
   fig = plt.figure()
   ax=fig.add_subplot(1, 2, 1)
   bx=fig.add_subplot(1, 2, 2)

   fontP = FontProperties()
   fontP.set_size('small')

   colors = np.random.rand(len(labels),3)

   for  c,i, target_name in zip(colors,range(len(labels)), target_names):
       ax.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c,
                  label=target_name,cmap=plt.cm.coolwarm)
       ax.legend(loc='upper center', bbox_to_anchor=(1.05, -0.05),
                 fancybox=True,shadow=True, ncol=len(labels),prop=fontP)
       ax.set_title('PCA')
       ax.tick_params(axis='both', which='major', labelsize=6)

   for c,i, target_name in zip(colors,range(len(labels)), target_names):
       bx.scatter(X_r2[y == i, 0], X_r2[y == i, 1], c=c,
                  label=target_name,cmap=plt.cm.coolwarm)
       bx.set_title('LDA');
       bx.tick_params(axis='both', which='major', labelsize=6)

   # Encode image to png in base64
   io = StringIO()
   fig.savefig(io, format='png')
   data = io.getvalue().encode('base64')

   return html.format(data)

run(host='localhost', port=8079, debug=True)
