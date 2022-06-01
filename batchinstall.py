#batchinstall.py
import os
libs = {'gym==0.10.5', 'grpcio==1.36.1', 'pyglet==1.4.10',\
        'tensorflow==1.8.0','numpy==1.14.5'}

try:
    for lib in libs:
        os.system('pip install ' + lib)
    print(lib + 'installed successful.')
except:
    print(lib + 'installed failed.')