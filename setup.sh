#!/bin/bash

pip3 install -r requirements.txt

### TODO, set the environment variables permanently
PROJECT_PATH=$(realpath ./)
echo $PROJECT_PATH
export PYTHONPATH=$PROJECT_PATH:$PROJECT_PATH/3rdparty/tenset/scripts:$PYTHONPATH

python3 -c "import dpro"
if [ $? = "1" ]; then
    cd ..
    git clone --recursive https://github.com/joapolarbear/dpro.git && cd dpro && bash setup.sh 
    cd $PROJECT_PATH
else
    echo "'dpro' has been isntalled."
fi

python3 -c "import optuna"
if [ $? = "1" ]; then
    cd .. && git clone --recursive https://github.com/joapolarbear/optuna
    cd optuna && python3 setup.py install
    cd $PROJECT_PATH
else
    echo "'optuna' has been isntalled."
fi

mkdir -p .workspace/ast_ansor && cp configs/ast_ansor/cfg.yaml .workspace/ast_ansor
mkdir -p tmp && cp configs/model/search_trial_20221119_1575* tmp/
alias cdmpp="bash $PROJECT_PATH/scripts/end2end/end2end_entry.sh"