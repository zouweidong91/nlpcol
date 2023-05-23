# vscode debug工具
# export REMOTE_DEBUG=True
# set REMOTE_DEBUG=True
import os, sys

def waitRemoteDebug():

    import ptvsd
    print('debug')
    address = ('0.0.0.0', 5678)
    ptvsd.enable_attach(address)
    ptvsd.wait_for_attach()
    print('### Connected Remote Debug ###')
