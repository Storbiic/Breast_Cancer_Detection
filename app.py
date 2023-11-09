import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import webbrowser

app = Flask(__name__)
model = pickle.load(open('BCD project\\.ipynb_checkpoints\modelSVM.pkl','rb'))

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
  input_features = [int(x) for x in request.form.values()]
  features_value = [np.array(input_features)]

  features_name = ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
       'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
       'bland_chromatin', 'normal_nucleoli']

  df = pd.DataFrame(features_value)
  output = model.predict(df)

  if output == 4:
      res_val = "tumeur  maligne (cancereuse)"
  else:
      res_val = "tumeur bénigne (non cancereuse)"


  return render_template('index.html', prediction_text='Patient a une {}'.format(res_val))

if __name__ == "__main__":
  webbrowser.open('http://127.0.0.1:5000', new=2)
  app.run()