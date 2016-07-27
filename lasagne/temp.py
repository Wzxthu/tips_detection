import numpy as np
import six.moves.cPickle as pickle

# Load the dataset
f_Xdata = open('data_n4.save', 'rb')
f_Ydata = open('label_n4.save', 'rb')
X_data = pickle.load(f_Xdata)
Y_data= pickle.load(f_Ydata)
f_Xdata.close()
f_Ydata.close()
f_Xdata = open('data_n1.save', 'wb')
f_Ydata = open('label_n1.save', 'wb')
np.save(f_Xdata, X_data)
np.save(f_Ydata, Y_data)
f_Xdata.close()
f_Ydata.close()

# Load the dataset
f_Xdata = open('data_n5.save', 'rb')
f_Ydata = open('label_n5.save', 'rb')
X_data = pickle.load(f_Xdata)
Y_data= pickle.load(f_Ydata)
f_Xdata.close()
f_Ydata.close()
f_Xdata = open('data_n2.save', 'wb')
f_Ydata = open('label_n2.save', 'wb')
np.save(f_Xdata, X_data)
np.save(f_Ydata, Y_data)
f_Xdata.close()
f_Ydata.close()