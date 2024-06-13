# autogluon-test-iris

This repository includes two sample projects:

 - /iris_project/ - and example of how to use the autogluon framework in python to create predictive models
 - /baseball_project/ - a slightly more advanced example of how to use the autogluon framework to create predictive models

___

#### Iris data set & code

I recommend starting with the iris_project.  Make 5-minutes to read about [the iris data set on wikipedia.](https://en.wikipedia.org/wiki/Iris_flower_data_set) After that, briefly look over notes about the [iris data set on the UC Irvine website.](https://archive.ics.uci.edu/dataset/53/iris)

In the case of the iris data set, our objective is to create a predictive model so that given different quantitative characteristics about different iris flowers, we can predict which iris species a given observation is about.  The inputs (x variables) also so-called independent variables are the measurements of (sepal length, sepal width, petal length, petal width).  The output (y variable) also called "depedent" variable is the class/category name of the flower, in this specific case it should be one of these values [Iris Setosa, Iris Versicolour, or Iris Virginica].

The data for this sub-project is contained in /data/iris_train.csv and /data/iris_test.csv

The code uses the python-based [autogluon framework](https://auto.gluon.ai/stable/index.html) which simplifies making machine learning models like Random Forests.

#### Description of code in /iris_project/

**01_build_autogluon_model.py** - this file loads training data (line 6) used to build a model, test data (line 7) used to evaluate the new model, and creates a new predictive model (line 12).  Autogluon automatically tries a bunch of different model types including two types of random forests, a neural network, and gradient boosted trees.  More details about what the different predictive model types are is included in the markdown file [model_descriptions.md](model_descriptions.md).

**02_evaluate_model.py** - this file shows how to print out the performance of different model types and also creates a "model leaderboard" that ranks the quality of the different models produced.

**03_use_model_for_prediction.py** - this file shows how to use a trained predictive model to make a prediction for a single input value (e.g., if we have the quantitative attributes of a flower, predict which species/category of flower that it is).

**04_load_specific_model_and_predict.py** - this file is similar to (03_use_model_for_prediction.py) but on line 52 it configures the predictor object to use a specific named model (in this case 'RandomForestGini' which is defined on line 6).


#### baseball data set & code

Instead of data on flowers, this project includes attributes of different baseball players.  There are multiple columns that are "dependent" values (Y variables) that relate to players having a specific position.  Rich L can provide additional details on the data.

The data for this sub-project is contained in /data/raw_data.csv and /data/raw_data_test.csv

#### Description of code in /baseball_project/

The code is organized similar to the iris project.

**00_unzip_data.py** - this file unzips the compressed data found in /data/raw_data.csv.gz

**01_build_autogluon_model.py** - this file loads training data (line 6) used to build a model, and drops some columns that we do not want to be used for analysis (lines 9-10 and lines 13-15).  Autogluon automatically tries a bunch of different model types including two types of random forests, a neural network, and gradient boosted trees.  More details about what the different predictive model types are is included in the markdown file [model_descriptions.md](model_descriptions.md).

**02_evaluate_model.py** - this file shows how to print out the performance of different model types and also creates a "model leaderboard" that ranks the quality of the different models produced.

**03_use_model_for_prediction.py** - this file shows how to use a trained predictive model to make a prediction for a single input value (e.g., if we have the quantitative attributes of a flower, predict which species/category of flower that it is).

**04_load_specific_model_and_predict.py** - this file is similar to (03_use_model_for_prediction.py) but on line 52 it configures the predictor object to use a specific named model (in this case 'RandomForestGini' which is defined on line 6).

___

## installing autogluon pre-requisited on macos

Getting autogluon setup on a Mac is kind of a pain.  I recommend running it on Linux or Windows if possible.  But if you want to try out the code on a mac, here are some installation instructions that may help.

### Uninstall "LLVM OpenMP library" (aka "libomp") if it was previous installed
brew uninstall -f libomp

### install specific version of libomp
wget https://raw.githubusercontent.com/Homebrew/homebrew-core/fb8323f2b170bd4ae97e1bac9bf3e2983af3fdb0/Formula/libomp.rb

### install libomp
brew install libomp.rb

### remove the installer script
rm libomp.rb

See also: https://auto.gluon.ai/stable/install.html

### installing autogluon

python3 -m pip install -r requirements.txt

