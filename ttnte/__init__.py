# Determine if display is terminal or notebook
try:
    import IPython

    IS_NOTEBOOK = "Terminal" not in IPython.get_ipython().__class__.__name__

except (NameError, ImportError):
    IS_NOTEBOOK = False

# This should always be the last line of this file
__version__ = "0.0.0b0"
