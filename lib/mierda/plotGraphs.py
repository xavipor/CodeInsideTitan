import matplotlib.pyplot as plt
import numpy as np
import pickle

def load_from_file(filename):
    """ Load object from file
    """
    object = []
    f = open(filename + '.pckl', 'rb')
    object = pickle.load(f)
    f.close()
    return object



errTSGD_300_001 = load_from_file('/home/javier/Documents/CODIGOTITANBUENO/CodeInsideTitan/lib/ExperimentsFirstNet/SGDFileToPlotCostT_0.001_300')
errVSGD_300_001 = load_from_file('/home/javier/Documents/CODIGOTITANBUENO/CodeInsideTitan/lib/ExperimentsFirstNet/SGDFileToPlotCostE_0.001_300')
accTSGD_300_001 = load_from_file('/home/javier/Documents/CODIGOTITANBUENO/CodeInsideTitan/lib/ExperimentsFirstNet/SGDFileToPlotEvalT_0.001_300')
accVSGD_300_001 = load_from_file('/home/javier/Documents/CODIGOTITANBUENO/CodeInsideTitan/lib/ExperimentsFirstNet/SGDFileToPlotEvalE_0.001_300')

errTSGD_300_03= load_from_file('/home/javier/Documents/CODIGOTITANBUENO/CodeInsideTitan/lib/ExperimentsFirstNet/SGDFileToPlotCostT_0.03_300')
errVSGD_300_03= load_from_file('/home/javier/Documents/CODIGOTITANBUENO/CodeInsideTitan/lib/ExperimentsFirstNet/SGDFileToPlotCostE_0.03_300')
accTSGD_300_03=	load_from_file('/home/javier/Documents/CODIGOTITANBUENO/CodeInsideTitan/lib/ExperimentsFirstNet/SGDFileToPlotEvalT_0.03_300')
accVSGD_300_03= load_from_file('/home/javier/Documents/CODIGOTITANBUENO/CodeInsideTitan/lib/ExperimentsFirstNet/SGDFileToPlotEvalE_0.03_300')


fig_1 = plt.figure(1,figsize=(200,300))
ax_1 = fig_1.add_subplot(1, 2, 1)
ax_2 = fig_1.add_subplot(1, 2, 2)

l = sorted(errTSGD_300_001.items())
x,y = zip(*l)
ax_1.plot(x,y,label="Error on Training LR = 0.001")

l = sorted(errVSGD_300_001.items())
x,y = zip(*l)
ax_1.plot(x,y,label="Error on Validation LR = 0.001")

l = sorted(errTSGD_300_03.items())
x,y = zip(*l)
ax_1.plot(x,y,label="Error on Training LR = 0.03")

l = sorted(errVSGD_300_03.items())
x,y = zip(*l)
ax_1.plot(x,y,label="Error on Validation LR = 0.03")


l = sorted(accTSGD_300_001.items())
x,y = zip(*l)
ax_2.plot(x,y,label="Accuracy on Training LR = 0.001")

l = sorted(accVSGD_300_001.items())
x,y = zip(*l)
ax_2.plot(x,y,label="Accuracy on Validation LR = 0.001")

l = sorted(accTSGD_300_03.items())
x,y = zip(*l)
ax_2.plot(x,y,label="Accuracy on Training LR = 0.03")

l = sorted(accVSGD_300_03.items())
x,y = zip(*l)
ax_2.plot(x,y,label="Accuracy on Validation LR = 0.03")

ax_1.set_title("Error on validation and training set with SGD with different Learning Rates")
ax_2.set_title("Accuracy on validation and training set with SGD with different Learning Rates")
ax_1.set_ylabel ("value")
ax_1.set_xlabel ("epochs")
ax_1.legend()
ax_2.set_ylabel ("value")
ax_2.set_xlabel ("epochs")
ax_2.legend()




errTSGD_300_001 = load_from_file('/home/javier/Documents/CODIGOTITANBUENO/CodeInsideTitan/lib/ExperimentsFirstNet/FileToPlotCostT_0.001_100')
errVSGD_300_001 = load_from_file('/home/javier/Documents/CODIGOTITANBUENO/CodeInsideTitan/lib/ExperimentsFirstNet/FileToPlotCostE_0.001_100')
accTSGD_300_001 = load_from_file('/home/javier/Documents/CODIGOTITANBUENO/CodeInsideTitan/lib/ExperimentsFirstNet/FileToPlotEvalT_0.001_100')
accVSGD_300_001 = load_from_file('/home/javier/Documents/CODIGOTITANBUENO/CodeInsideTitan/lib/ExperimentsFirstNet/FileToPlotEvalE_0.001_100')

errTSGD_300_03= load_from_file('/home/javier/Documents/CODIGOTITANBUENO/CodeInsideTitan/lib/ExperimentsFirstNet/FileToPlotCostT_0.03_100')
errVSGD_300_03= load_from_file('/home/javier/Documents/CODIGOTITANBUENO/CodeInsideTitan/lib/ExperimentsFirstNet/FileToPlotCostE_0.03_100')
accTSGD_300_03=	load_from_file('/home/javier/Documents/CODIGOTITANBUENO/CodeInsideTitan/lib/ExperimentsFirstNet/FileToPlotEvalT_0.03_100')
accVSGD_300_03= load_from_file('/home/javier/Documents/CODIGOTITANBUENO/CodeInsideTitan/lib/ExperimentsFirstNet/FileToPlotEvalE_0.03_100')


fig_2 = plt.figure(2,figsize=(200,300))
ax_1 = fig_2.add_subplot(1, 2, 1)
ax_2 = fig_2.add_subplot(1, 2, 2)

l = sorted(errTSGD_300_001.items())
x,y = zip(*l)
ax_1.plot(x,y,label="Error on Training LR = 0.001")

l = sorted(errVSGD_300_001.items())
x,y = zip(*l)
ax_1.plot(x,y,label="Error on Validation LR = 0.001")

l = sorted(errTSGD_300_03.items())
x,y = zip(*l)
ax_1.plot(x,y,label="Error on Training LR = 0.03")

l = sorted(errVSGD_300_03.items())
x,y = zip(*l)
ax_1.plot(x,y,label="Error on Validation LR = 0.03")


l = sorted(accTSGD_300_001.items())
x,y = zip(*l)
ax_2.plot(x,y,label="Accuracy on Training LR = 0.001")

l = sorted(accVSGD_300_001.items())
x,y = zip(*l)
ax_2.plot(x,y,label="Accuracy on Validation LR = 0.001")

l = sorted(accTSGD_300_03.items())
x,y = zip(*l)
ax_2.plot(x,y,label="Accuracy on Training LR = 0.03")

l = sorted(accVSGD_300_03.items())
x,y = zip(*l)
ax_2.plot(x,y,label="Accuracy on Validation LR = 0.03")

ax_1.set_title("Error on validation and training set with Adam with different Learning Rates")
ax_2.set_title("Accuracy on validation and training set with Adam with different Learning Rates")
ax_1.set_ylabel ("value")
ax_1.set_xlabel ("epochs")
ax_1.legend()
ax_2.set_ylabel ("value")
ax_2.set_xlabel ("epochs")
ax_2.legend()
plt.show()

