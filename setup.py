from setuptools import setup
from setuptools import find_packages


setup(name='pyCIFA',
      version='0.2',
      description='Python translation of original matlab implementation of CIFA',
      classifiers=[
        #'Development Status :: 3 - Alpha',
        #'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Signal processing :: Machine learning',
      ],
      keywords='CIFA COBE PCOBE COBEC NMF',
      url='https://bitbucket.org/kharyuk/pycifa/',
      license='',
      packages=find_packages(exclude=('*construct_w', '*gnmf', '*jive', '*mcca', '*metrics', '*mmc_nn', '*pmf_sobi')),
      install_requires=[
          'numpy',
          'scipy'
      ],
      python_requires='==2.7',
      zip_safe=False)
