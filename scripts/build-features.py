from os import listdir
from importlib import import_module

feature_root = 'scripts/features'

for filename in listdir(feature_root):
    if filename.endswith('.py'):
        print('Running ', filename)
        import_module('.' + filename[:-3], 'features').main()