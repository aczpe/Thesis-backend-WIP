################################ WORK IN PROGRESS #################################
import os
import json
from pyspark.sql import functions as F # pl a maxok osszekeverese ellen
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, BooleanType #a schema miatt
from pyspark.sql.window import Window #for df joining,might replace it later
#these need distutils which has been removed from python 3.12, fix: installed setuptools with pip
from pyspark.ml.feature import VectorAssembler,StringIndexer #for logistic regression and calculating correlation,string to num conversions
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.stat import Correlation #for correlation analysis

################################ TODO: fajlokba kiszervezni #################################

#might be necessary if backend is installed on a different machine
os.environ["SPARK_HOME"] = "C:\\apps\\spark-3.5.1-bin-hadoop3"
os.environ["JAVA_HOME"] = "C:\\jdk\\Java\\jdk-1.8"
os.environ["PYSPARK_HOME"] = "C:\\Users\\Sajat\\AppData\\Local\\Programs\\Python\\Python312\\python.exe"

spark = SparkSession.builder.appName("Gladiator_app").getOrCreate()

sc = spark.sparkContext

csv_path = "D:\\egyetem\\szakdoga\\gladiator_data.csv"

df = spark.read.csv(csv_path, header=True, inferSchema = True)
# droppolunk mindent amire a gladiator azt mondta nem relevans
df = df.drop('Birth Year','Origin','Patron Wealth','Public Favor','Allegiance Network','Psychological Profile','Personal Motivation','Previous Occupation' , 'Battle Strategy', 'Social Standing', 'Crowd Appeal Techniques', 'Weapon Of Choice' ) 
#TODO: nanok kiszedese, adattisztitas

#tisztan numerikus df elkészitése for ml and some charts
converted_df = df

indexer = StringIndexer(inputCol='Special Skills',outputCol='Special Skills num')
converted_df = indexer.fit(converted_df).transform(converted_df)

indexer = StringIndexer(inputCol='Equipment Quality',outputCol='Equipment Quality num')
converted_df = indexer.fit(converted_df).transform(converted_df)

indexer = StringIndexer(inputCol='Injury History',outputCol='Injury History num')
converted_df = indexer.fit(converted_df).transform(converted_df)

indexer = StringIndexer(inputCol='Diet and Nutrition',outputCol='Diet and Nutrition num')
converted_df = indexer.fit(converted_df).transform(converted_df)

indexer = StringIndexer(inputCol='Tactical Knowledge',outputCol='Tactical Knowledge num')
converted_df = indexer.fit(converted_df).transform(converted_df)

indexer = StringIndexer(inputCol='Health Status',outputCol='Health Status num')
converted_df = indexer.fit(converted_df).transform(converted_df)

indexer = StringIndexer(inputCol='Training Intensity',outputCol='Training Intensity num')
converted_df = indexer.fit(converted_df).transform(converted_df)

converted_df = converted_df.withColumn("Survived", converted_df["Survived"].cast("integer"))

#TODO: kategoriara szurni kell meg 
converted_df = converted_df.drop('Special Skills', 'Equipment Quality', 'Injury History', 'Diet and Nutrition', 'Tactical Knowledge',
                   'Health Status', 'Training Intensity')

#category: string
def minmax_to_json(category):
    result ={ 
    "max" : calc_max(category).collect()[0][0], 
    "min" : calc_min(category).collect()[0][0], 
    "count" : calc_count(category).collect()[0][0], 
    "samples" : df.select(category).sample(withReplacement=True, fraction=0.0013).collect()
    }

    with open("D:\\egyetem\\szakdoga\\output.json", "w") as outfile: 
        json.dump(result, outfile)
    

def meanmedianmode_to_json(category):
    result ={ 
    "mean" : calc_mean(category).collect()[0][0], 
    "median" : calc_median(category).collect()[0][0],
    "mode" : calc_mode(category).collect()[0][0], 
    "count" : calc_count(category).collect()[0][0], 
    "samples" : df.select(category).sample(withReplacement=True, fraction=0.0013).collect()
    }

    with open("D:\\egyetem\\szakdoga\\output.json", "w") as outfile: 
        json.dump(result, outfile)

def std_to_json(category):
    result ={ 
    "std" : calc_std(category).collect()[0][0], 
    "range" : calc_range(category).collect()[0][0], 
    "count" : calc_count(category).collect()[0][0], 
    "samples" : df.select(category).sample(withReplacement=True, fraction=0.0013).collect()
    }

    with open("D:\\egyetem\\szakdoga\\output.json", "w") as outfile: 
        json.dump(result, outfile)

def frequency_to_json(category):
    result = {
        "counts" : calc_frequency_distribution(category)
    }

    with open("D:\\egyetem\\szakdoga\\output.json", "w") as outfile: 
        json.dump(result, outfile)

#def match_results_to_json(results):
#    result = {
#        "opponent" : str
#        "winner" :str
#        "likelihood":int
#    }

def new_gladiator_to_json(attributes):
    result = {
        "name_taken" : new_gladiator(attributes)
    }
    with open("D:\\egyetem\\szakdoga\\output.json", "w") as outfile: 
        json.dump(result, outfile)


def correlation_to_json(categories,corr_method):
    result = {
        "correlation" : correlation(categories,corr_method)
    }
    with open("D:\\egyetem\\szakdoga\\output.json", "w") as outfile: 
        json.dump(result, outfile)

def calc_mean(category):
    #TODO: a kliens szurje ki a stringtypeokat
    return df.agg(F.mean(category))

def calc_median(category):
    #TODO: a kliens szurje ki a stringtypeokat
    return df.agg(F.median(category))

def calc_min(category):
    #TODO: a kliens szurje ki a stringtypeokat
    return df.agg(F.min(category))

def calc_max(category):
    #TODO: a kliens szurje ki a stringtypeokat
    return df.agg(F.max(category))

def calc_count(category):
    #TODO: a kliens szurje ki a stringtypeokat
    return df.agg(F.count(category))

#most common element
def calc_mode(category):
    #TODO: a kliens szurje ki a stringtypeokat
    return df.groupby(category).count().orderBy('count', ascending = False).first()[1]

#standard deviation
def calc_std(category):
    #TODO: a kliens szurje ki a stringtypeokat
    return df.agg(F.stddev(category))

def calc_range(category):
    #TODO: a kliens szurje ki a stringtypeokat
    return df.agg(F.max(category)-F.min(category))

#categories : list of category names
#corrmethod: the type of correlation eg. pearson
def correlation(categories, corrmethod):
    assembler = VectorAssembler(inputCols=categories, outputCol="features") #https://medium.com/@demrahayan/a-guide-to-correlation-analysis-in-pyspark-22824b9a5dda
    df_vectorized = assembler.transform(df.select(categories))
    corr_matrix = Correlation.corr(df_vectorized, "features", method=corrmethod)
    result = corr_matrix[0].toArray()

def calc_frequency_distribution(category):
    return df.groupBy(category).count()

#returns 0 if successfully admitted, otherwise returns 1
#attributes should to be a tuple
#TODO: hibakezeles rossz tipusu bemenetre
def new_gladiator(attributes):
    #TODO: kliens autofillelje az ures mezoket
    #return 1 if gladiator with the exact same attributes already present in the dataset
    if df.where(df.Name == attributes[0]).where(df.Height == attributes[4]).where(df.Weight == attributes[5]).count()>0:
        return 1
    else:
        new_row = spark.createDataFrame([attributes], df.columns)
        df = df.union(new_row)
        #TODO: kiirni a csvbe
        return 0

#searchbbar fuggveny nevkereseshez ide
#def

possible_pairings = {
    "Retiarius": ["Secutor"],
    "Secutor": ["Retiarius"],
    "Provocator": ["Provocator"],
    "Murmillo": ["Thraex","Hoplomachus"],
    "Hoplomachus": ["Thraex","Murmillo"],
    "Thraex": ["Thraex","Murmillo","Hoplomachus"]
}

#it finds the ideal opponent of the candidate using logistic regression
#returns a tuple of the opponents name, the predicted winner and the likeliness of their victory
#since names arent unique to gladiators we also haave to account for their weight and height
def matchmake(name,weight,height):
    
    #szures a megfelelo kategoriakra(kisebb dataset, ellenfel kivalasztasanal nincs folosleges parsing de eleg nagy a traininghez)
    #inspiracio: https://www.youtube.com/watch?v=YpI4_RrargQ&t=850s
    category = converted_df.where(converted_df.Name == name).where(converted_df.Height == height).where(converted_df.Weight == weight).first()['Category']
    matchmaker_df = converted_df.filter(converted_df.Category.isin(possible_pairings[category]))
    
    #ai training
    assembler = VectorAssembler(inputCols=['Age','Height','Weight','Wins','Losses','Mental Resilience','Battle Experience'
                                           ,'Special Skills num', 'Equipment Quality num','Injury History num', 'Training Intensity num'],
                                           outputCol="features")
    output = assembler.transform(matchmaker_df)

    model_df = output.select('features','Survived')

    training_df, test_df = model_df.randomSplit([0.7, 0.3])

    log_reg = LogisticRegression(labelCol='Survived').fit(training_df)

    train_results = log_reg.evaluate(training_df).predictions
    results = log_reg.evaluate(test_df).predictions

    #tulelesi ratak hozzaadasa a matchmaker_dfhez   
    probability = results.select('probability')

    ##source:https://www.statology.org/pyspark-add-column-from-another-dataframe/
    #w = Window().orderBy(F.lit('A'))
    #matchmaker_df = matchmaker_df.withColumn('id', F.row_number().over(w))
    #probability = probability.withColumn('id', F.row_number().over(w))
    #matchmaker_df = matchmaker_df.join(probability, on=['id']).drop('id')
#
    #matchmaker_df.show(10)
#
    #candidate_likelyhood = matchmaker_df.where((matchmaker_df.Name == name) & (matchmaker_df.Weight == weight) & (matchmaker_df.Height == height)).first()['Death Likelihood']
    #
    ##valaszott gladiator kivetele a mintak kozul(hogy ne keruljon magaval parositasba)
    #matchmaker_df = matchmaker_df.where((matchmaker_df.Name != name) & (matchmaker_df.Weight != weight) & (matchmaker_df.Height != height))
    #
    ##megfelelo ellenfel megtalalasa
    #opponent = matchmaker_df.agg(F.min(matchmaker_df.select('Death Likelihood')-candidate_likelyhood)).collect()[0][0]
    #
    ##vegso tuple

#calc_mean("Age")
#BUG: ua. a fv-t tobbszor is le lehet futtatni, de ha masikat valasztok, elso alkalomra jol felulirja viszont nosuchfileexceptiont dob
#latszolag random idokozonkent tortenik

#minmax_to_json("Age")

matchmake('Cosconius Hostius',54,176)

input("Press Enter to continue")

sc.stop()