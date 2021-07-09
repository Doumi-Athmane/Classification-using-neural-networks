#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pyspark')


# In[17]:


from pyspark.sql import SparkSession

spark = SparkSession        .builder        .appName("TP_BDM-Doumi_Boudis")     .config("spark.some.config.option", "some-value")     .getOrCreate()


# In[3]:


dataset= spark.read.csv('ad.data',header=False,inferSchema=True)


# In[4]:


from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col


# In[20]:


print(dataset.dtypes)


# In[5]:


from pyspark.sql.types import IntegerType

dataset = dataset.withColumn("_c0", col("_c0").cast(IntegerType()))
dataset = dataset.withColumn("_c1", col("_c1").cast(IntegerType()))
dataset = dataset.withColumn("_c2", col("_c2").cast(IntegerType()))
dataset = dataset.withColumn("_c3", col("_c3").cast(IntegerType()))


# In[28]:


print(dataset.dtypes)


# In[6]:


dataset = dataset.fillna(0)


# In[7]:


modele_couche1 = [1558,500,100,2]
modele_couche2 = [1558,100,50,2]
modele_couche3 =[1558,1000,500,100,2]
modele_couche4 =[1558, 800, 400,200,2]
trainer = MultilayerPerceptronClassifier()


# In[9]:


string_columns= [item[0] for item in dataset.dtypes if item[1].startswith(
    'string')] #Ici on sais bien que seulement la derniere colonne est de type string
indexer = StringIndexer(inputCol=string_columns[0] , outputCol="label") 
dataset = indexer.fit(dataset).transform(dataset) 
numeric_columns = [item[0] for item in dataset.dtypes if item[1].startswith(
    'int')]

featuresCreator = VectorAssembler(
    inputCols=numeric_columns,
    outputCol='features')
adData = featuresCreator.transform(dataset).select(col("features"),col("label"))


# In[9]:


paramGrid = ParamGridBuilder().addGrid(trainer.maxIter,[5,10,20])            .addGrid(trainer.layers,[modele_couche1,modele_couche2,modele_couche3,modele_couche4])            .build()

crossValidator = CrossValidator(estimator=trainer,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(),
                          numFolds=3)


# In[10]:


model = crossValidator.fit(adData)


# In[11]:


model.bestModel.getLayers()


# In[12]:


model.bestModel.getMaxIter()


# In[13]:


classifier = MultilayerPerceptronClassifier(labelCol='label',
                                            featuresCol='features',
                                            maxIter=20,
                                            layers=modele_couche1,
                                            blockSize=128,
                                            seed=1234)


# In[14]:


bestModel = classifier.fit(adData)


# In[15]:


predictions = bestModel.transform(adData)


# In[16]:


evaluator = MulticlassClassificationEvaluator(labelCol="label")
evaluator.evaluate(predictions)


# In[69]:


import pandas as pd
import numpy as np

df = dataset.toPandas()        
df.head()


# In[23]:


cor = df.corr()
cor_target = abs(cor["label"])
relevant_features = cor_target[cor_target>0.3]
relevant_features


# In[52]:


for columnn in df :
    if ((columnn not in relevant_features) and (columnn != "_c1558")):
         df = df.drop(columnn,1) 
df
    


# In[71]:


newDataset=spark.createDataFrame(df) 
newDataset


# In[60]:



attributs=newDataset.schema.names[0:28]
typee = newDataset.schema.names[28]
#indexer = StringIndexer(inputCol=typee , outputCol="label") 
#newDataset = indexer.fit(newDataset).transform(newDataset) 
featuresCreator = VectorAssembler(
    inputCols=attributs,
    outputCol='features')
newAdData = featuresCreator.transform(newDataset).select(col("features"),col("label"))


# In[61]:


new_modele_couche = [28,500,100,2]
classifier = MultilayerPerceptronClassifier(labelCol='label',
                                            featuresCol='features',
                                            maxIter=2,
                                            layers=new_modele_couche,
                                            blockSize=128,
                                            seed=1234)


# In[62]:


bestModel_bestAttribut = classifier.fit(newAdData)


# In[ ]:


newPrediction = best_model_with_new_dataset.transform(newAdData)
evaluator = MulticlassClassificationEvaluator(labelCol="label")
evaluator.evaluate(newPrediction)

