#!/usr/bin/env python
# encoding: utf-8

from argparse import ArgumentParser
from DeepJetCore.DataCollection import DataCollection

parser = ArgumentParser('convert a data collection to a single set of numpy arrays. Warning, this can produce a large output')
parser.add_argument('inputDataCollection')
parser.add_argument('outputFilePrefix')
parser.add_argument('--isplit', default=0, type=int,
                    help='do ith split')
parser.add_argument('--maxsplit', default=1, type=int,
                    help='split into maxsplit data sets')
args = parser.parse_args()

print('reading data collection')

dc=DataCollection()
dc.readFromFile(args.inputDataCollection)

print('producing feature array')
feat=dc.getAllFeatures(isplit = args.isplit, maxsplit = args.maxsplit)

print('producing truth array')
truth=dc.getAllLabels(isplit = args.isplit, maxsplit = args.maxsplit)

print('producing weight array')
weight=dc.getAllWeights(isplit = args.isplit, maxsplit = args.maxsplit)

print('producing spectator array')
spectator=dc.getAllSpectators()

print('producing means and norms array')
means=dc.means

from numpy import save

print('saving output')
for i in range(len(feat)):
    save(args.outputFilePrefix+'_features_'+str(i) +'.npy', feat[i])
    
for i in range(len(truth)):
    save(args.outputFilePrefix+'_truth_'+str(i) +'.npy', truth[i])
    
for i in range(len(weight)):
    save(args.outputFilePrefix+'_weights_'+str(i) +'.npy', weight[i])

for i in range(len(spectator)):
    save(args.outputFilePrefix+'_spectators_'+str(i) +'.npy', weight[i])
    
save(args.outputFilePrefix+'_meansandnorms.npy', means)
