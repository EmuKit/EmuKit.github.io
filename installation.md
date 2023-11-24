---
layout: default
title: Installation
permalink: /installation/
---

<h1>Installation</h1>

Installation from sources and using pip are both supported.

#### Dependencies / Prerequisites
Emukit requires Python 3.7 or above, NumPy and SciPy for basic functionality. Some core features also need [GPy](https://sheffieldml.github.io/GPy/). Some advanced elements may have their own dependencies, but their installation is optional.

Required dependecies can be installed from the [requirements](https://github.com/emukit/emukit/blob/main/requirements/requirements.txt) file via 

```
pip install -r requirements/requirements.txt
```

#### Install using pip 
Just write:

```
pip install emukit
```


#### Install from sources
To install Emukit from source, create a local folder where you would like to put Emukit source code, and run following commands:

```
git clone https://github.com/emukit/emukit.git
cd emukit
python setup.py install
```

Alternatively you can run

```
pip install git+https://github.com/emukit/emukit.git
```

