#
# Alunos: Edson / Kariston / Anderson Salata
#

from __future__ import print_function

import sys
from operator import add

from pyspark.sql import SparkSession

print("************************************", '\n')
print("Parametro <file>: ", sys.argv[1], '\n')
palavra_excluida = sys.argv[2]
print("Parametro <palavra_chave>: ", palavra_excluida, '\n')
print("************************************", '\n')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: wordcount <file>", file=sys.stderr)
        sys.exit(-1)

    spark = SparkSession\
        .builder\
        .appName("PythonWordCount")\
        .getOrCreate()

    lines = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0])
    counts = lines.flatMap(lambda x: x.split(' ')) \
                  .filter(lambda x: palavra_excluida not in x.lower()) \
                  .map(lambda x: (x, 1)) \
                  .reduceByKey(add)
    output = counts.collect()
    for (word, count) in output:
        print("%s: %i" % (word, count))

    spark.stop()
