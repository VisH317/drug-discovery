mkdir -p data
mkdir -p data/tox21
wget http://bioinf.jku.at/research/DeepTox/tox21_dense_train.csv.gz -P data/tox21
wget http://bioinf.jku.at/research/DeepTox/tox21_labels_train.csv.gz -P data/tox21
wget http://bioinf.jku.at/research/DeepTox/tox21_dense_test.csv.gz -P data/tox21
wget http://bioinf.jku.at/research/DeepTox/tox21_labels_test.csv.gz -P data/tox21
gzip -d data/tox21/tox21_dense_train.csv.gz
gzip -d data/tox21/tox21_labels_train.csv.gz
gzip -d data/tox21/tox21_dense_test.csv.gz
gzip -d data/tox21/tox21_labels_test.csv.gz