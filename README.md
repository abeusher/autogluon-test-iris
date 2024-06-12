# autogluon-test-iris

## installing autogluon pre-requisited on macos


# Uninstall "LLVM OpenMP library" (aka "libomp") if it was previous installed
brew uninstall -f libomp

# install specific version of libomp
wget https://raw.githubusercontent.com/Homebrew/homebrew-core/fb8323f2b170bd4ae97e1bac9bf3e2983af3fdb0/Formula/libomp.rb

# install libomp
brew install libomp.rb

# remove the installer script
rm libomp.rb

See also: https://auto.gluon.ai/stable/install.html

## installing autogluon

python3 -m pip install -r requirements.txt


## Description of files:

 - 01_make_train_and_test_data.py - splits input "IRIS.csv" into two separate files, one for training and one for testing
 - 02_run_autogluon_classifier.py - creates multiple predictive models using autogluon