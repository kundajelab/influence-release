from distutils.core import setup

if __name__== '__main__':
    setup(include_package_data=True,
          description='Influence functions',
          url='NA',
          download_url='NA',
          version='0.1',
          packages=['influence'],
          setup_requires=[],
          install_requires=['numpy>=1.9', 'tensorflow>=1.0'],
          scripts=[],
          name='influence')
