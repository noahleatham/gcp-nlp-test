# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 12:07:11 2021

@author: NoahLeatham
"""

import sys
from sparknlp.annotator import Lemmatizer, Stemmer, Tokenizer, Normalizer
from sparknlp.base import DocumentAssembler, Finisher
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, LongType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.clustering import LDA
from pyspark.sql.functions import col, lit, concat
from pyspark.sql.utils import AnalysisException

# Find bucket name
try:
    bucket = sys.argv[1]
except IndexError:
    print("No bucket name")
    sys.exit(1)

# Create sparksession to load data
spark = SparkSession.builder.appName("ml topic model").getOrCreate()

# Create spark schema
fields = [StructField("source_id", LongType(), True),
          StructField("year", LongType(), True),
          StructField("title", StringType(), True),
          StructField("abstract", StringType(), True),
          StructField("full_text", StringType(), True)]
schema = StructType(fields)

# Set bucket path
gs_path = f"gs://{bucket}/papers.csv"

# Load in csv data
try:
    text_data = (spark.read.format('csv')
                 .option('header','true')
                 .option('multiline','true')
                 .option('escape','"')
                 .option('wholeFile','true')
                 #.options(codec="org.apache.hadoop.io.compress.GzipCodec")
                 .schema(schema)
                 .load(gs_path)
                 )
except AnalysisException:
    print("Failed to read data, system exiting...")
    sys.exit(1)

# Transform data into ready-format
df_train = (
    text_data
    .fillna("")
    .select(
        concat(
            col("title"),
            lit(" "),
            col("abstract"),
            lit(" "),
            col("full_text")
        ).alias("text")
    )
)

# Create pipeline objects
document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token")
normalizer = Normalizer().setInputCols(["token"]).setOutputCol("normalizer")
stemmer = Stemmer().setInputCols(["normalizer"]).setOutputCol("stem")
finisher = Finisher().setInputCols(["stem"]).setOutputCols(["to_spark"]).setValueSplitSymbol(" ")
stopword_remover = StopWordsRemover(inputCol="to_spark", outputCol="filtered")
tf = CountVectorizer(inputCol="filtered", outputCol="raw_features")
idf = IDF(inputCol="raw_features", outputCol="features")
lda = LDA(k=10, maxIter=10)

# Create pipeline
pipeline = Pipeline(
    stages = [
        document_assembler,
        tokenizer,
        normalizer,
        stemmer,
        finisher,
        stopword_remover,
        tf,
        idf,
        lda
    ]
)

model = pipeline.fit(df_train)
vocab = model.stages[-3].vocabulary
raw_topics = model.stages[-1].describeTopics().collect()
topic_inds = [ind.termIndices for ind in raw_topics]

topics = []
for topic in topic_inds:
    _topic = []
    for ind in topic:
        _topic.append(vocab[ind])
    topics.append(_topic)

# Let's see our topics!
for i, topic in enumerate(topics, start=1):
    print(f"topic {i}: {topic}")