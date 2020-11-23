from distutils.core import setup

setup(name='gym_game_of_drones',
      version='0.0.1',
      description='Advanced gym environment and dataset collector game, built on top of Airsim Drone Racing Lab',
      author='Yann Bouteiller',
      author_email='yann.bouteiller@polymtl.ca',
      download_url='N/A',
      packages=['gym_game_of_drones', ],
      licence='MIT',
      install_requires=['pygame (>=2.0.0.dev6)',
                'zmq',
                'airsimdroneracinglab',
                'pyinstrument',
                'gym',
                'pathlib',
                'requests'])
