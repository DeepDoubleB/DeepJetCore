'''
Created on 21 Feb 2017

@author: jkiesele
'''

from DataCollection import DataCollection
from TrainData_deepCSV import TrainData_deepCSV
from TrainData_veryDeepJet import TrainData_veryDeepJet

dc=DataCollection()
#dc.convertListOfRootFiles('/Users/jkiesele/Cernbox/batchtest/samples.txt', TrainData_deepCSV(), '/Users/jkiesele/Cernbox/batchtest/deepCSV')

dc.convertListOfRootFiles('/Users/jkiesele/Cernbox/batchtest/samples.txt', TrainData_veryDeepJet(), '/Users/jkiesele/Cernbox/batchtest/deepJet')
