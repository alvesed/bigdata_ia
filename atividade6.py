#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
from operator import add

from pyspark.sql import SparkSession


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: atividade6 <file>", file=sys.stderr)
        sys.exit(-1)

    spark = SparkSession\
        .builder\
        .appName("Atividade6")\
        .getOrCreate()

    lines = spark.read.csv(sys.argv[1]).rdd
    
    # selecionando apenas a coluna do codigo da acomodacao
    code = lines.map(lambda l: l['_c0'])
    
    acc_id = code.take(code.count())
        
    total_objetos = 0
    
    for objeto in acc_id:
        total_objetos = total_objetos+1;
        
    print("numero de objetos: %s" % total_objetos)

    # apenas registros que sao "Entire room"
    entire_rooms = lines.filter(lambda l: l['_c8'] == "Entire home/apt")

    print("Numero de registros que sao acomodacoes inteiras: %s" % entire_rooms.count())

    spark.stop()
