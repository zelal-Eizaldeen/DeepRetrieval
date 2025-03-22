# WikiSQL Raw Data Process

References: https://github.com/salesforce/WikiSQL/tree/master

```shell
cd code/data/raw_data/wikisql

git clone https://github.com/salesforce/WikiSQL
cd WikiSQL
pip install -r requirements.txt
tar xvjf data.tar.bz2


# 1. create clean database
python create_database.py

# 2. get clean sql
```