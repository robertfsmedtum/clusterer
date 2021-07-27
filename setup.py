from setuptools import setup

setup(
  name = 'clusterer',
  version = '0.1.0',
  license='BSD',
  description = 'UMAP + HDBSCAN clusterer',
  author = 'Robert Kaczmarczyk',
  author_email = 'robertfsmedtum@gmail.com',
  keywords = [
    'clustering',
    'network analysis',
    'umap',
    'hdbscan'
  ],
  install_requires=[
      'numpy',
      'streamlit',
      'pandas',
      'umap-learn',
      'umap',
      'numba',
      'scipy',
      'hdbscan',
      'sklearn',
      'bokeh==2.2.2',
      'decorator==4.4.2',
      'networkx==2.5.1',
      'wordcloud'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Scientists',
    'Topic :: Scientific/Engineering :: Clustering',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 3.8.5',
  ],
)