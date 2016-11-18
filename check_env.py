"""
Script to check if the required packages for the workshop are installed 

Author: Bargava Subramanian

"""
import sys

# requirements
has = dict(
    IPython='5.0.0',
    matplotlib='1.5.1',   
    lightfm='1.9',
    nimfa='1.3.2',
    numpy='1.11.0',
    pandas='0.19',
    scipy='0.17.0',
    seaborn='0.7',
    sklearn='0.18'

)



returns = 0

# check installed packages
for module in has.keys():
    try:
        _module = module.split('-')[-1]
        __module__ = __import__(_module, globals(), locals(), [], 0)
        exec('%s = __module__' % _module)
    except ImportError:
        print("%s:: %s" % (module, sys.exc_info()[1]))
        #run.pop(module, None)
        returns += 1


# check required versions
from distutils.version import LooseVersion as V
for module,version in has.items():
    try:
        _module = module.split('-')[-1]
        assert V(eval(_module).__version__) >= V(version)
    except NameError:
        pass # failed import
    except AttributeError:
        pass # can't version-check non-standard packages...
    except AssertionError:
        print("%s:: Version >= %s is required" % (module, version))
        returns += 1


# final report
if not returns:
    print('-'*50)
    print('OK.  All required items installed.')

sys.exit(returns)



