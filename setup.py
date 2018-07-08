from setuptools import setup

setup(name='neuro_evolution',
      version='0.1',
      description='Neuro-evolution wrapper for NN hyperparameter tuning',
      url='https://github.com/subpath/neuro-evolution',
      author='Alexander Osipenko',
      author_email='subpath@ya.ru',
      license='MIT',
      packages=['neuro_evolution'],
      install_requires=['tqdm', 'numpy', 'logger', 'keras'],
      zip_safe=False)