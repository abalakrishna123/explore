import pickle 
import matplotlib.pyplot as plt

a = pickle.load( open( "q_learning_data1/results_copy.p", "rb" ) )
plt.plot((a['steps'][4000:5100]) )
plt.show()