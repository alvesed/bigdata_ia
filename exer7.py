# Edson exercicio 7
from __future__ import print_function

import sys
from operator import add

from pyspark.sql import SparkSession


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: atividade 7 <file>", file=sys.stderr)
        sys.exit(-1)

    spark = SparkSession\
        .builder\
        .appName("SparkApp")\
        .getOrCreate()

    lines = spark.read.csv(sys.argv[1]).rdd

    # selecionando apenas a coluna relativa a user rating score
    user_rs = lines.map(lambda l: l['_c5'] > '90')
    #inversor_gouped = lines.map(lambda l: (l['_c7'], l)).groupByKey()
    

    # buscando os classificados para audiencias acima de 17 anos?
    tv_ma = user_rs.filter(lambda r: r['_c1'] == 'TV_MA')
        
    tv_ma_list = tv_ma.mapValues(list)

    # salvando resultados em disco
    tv_ma.saveAsTextFile("over17.txt")
    tv_ma_list.saveAsTextFile("over17list.txt")

    spark.stop()
