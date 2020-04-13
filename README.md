# Robotic Manipulation Inside a Simulation Environment Through Supervised Learning

This repository makes it possible for researchers to easily gather a dataset of images for training a robot to perform robotic manipulation. The dataset contains 2-d images that have been rendered with the use of the V-REP physics simulation engine. The repository enables users to readily apply domain randomization through varying light conditions, object textures, background colors and textures, camera position, target object location and target object dimensions.
Additionally, the repository provides two different methods with which the robot can be trained to grasp the object using the previously collected dataset. The first method depends on classification, so the robot receives various potential grasps on the action space (the space in which it is allowed to perform grasps) as well as one spatial image and outputs the probability that the given grasps will result in a successful grasp. The second method depends on regression, so the robot has to learn to predict a successful grasp given just the spatial image. A grasp is a 3-dimensional vector that is made up of the x and y location on the action space as well as of the orientation of the grasp.  

## Getting Started

### Prerequisites

1. Linux Operating System, preferably Ubuntu 18.04.

2. Please [download the V-REP](https://coppeliarobotics.com/downloads) physics simulation engine. The repository was

   created using the V-REP PRO EDU, Ubuntu 18.04

3.

   ```
   sudo apt-get install libxkbcommon-x11-dev

   export PATH=$PATH:~/Qt/Tools/QtCreator/bin

   ```

4.  Create SWAP file, as the installation of torch is a highly memory consuming procedure.

    ```

    sudo fallocate -l 8G /swapfile

     sudo chmod 600 /swapfile

     sudo mkswap /swapfile

     sudo swapon /swapfile

     sudo cp /etc/fstab /etc/fstab.back

     echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

     ```

### Installation

1.  If you don't already have it, [install Python](https://www.python.org/downloads/).

    This repository was developed is compatible with Python 2.7, 3.3, 3.4, 3.5 and 3.6.

2.  General recommendation for Python development is to use a Virtual Environment.
    For more information, see https://docs.python.org/3/tutorial/venv.html

    Install and initialize the virtual environment with the "venv" module on Python 3 (you must install [virtualenv](https://pypi.python.org/pypi/virtualenv) for Python 2.7):

    ```
    python -m venv mytestenv # Might be "python3" or "py -3.6" depending on your Python installation
    cd mytestenv
    source bin/activate      
    ```

### Quickstart

1.  Clone the repository.

    ```
    git clone https://github.com/gthd/supervised_learning.git
    ```

2.  Install the dependencies using pip.

    ```
    cd supervised_learning
    pip install -r requirements.txt
    ```

## Demo

A demo app is included to show how to use the project.

To collect the dataset run:

1. `python supervised/supervised.py`

To train the robot using classification run:

2. `python supervised/q_network.py`

To train the robot using regression run:

3. `python supervised/regression.py`

## Contributing

Please read [CONTRIBUTING.md](Contributing.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* [**Georgios Theodorou**](https://github.com/gthd)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* I want to acknowledge the help and guidance I received from my supervisor [Edward Johns](https://www.imperial.ac.uk/people/e.johns).
