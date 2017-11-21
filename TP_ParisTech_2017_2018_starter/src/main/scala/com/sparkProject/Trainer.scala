package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.mllib.regression.LabeledPoint
object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /********************************************************************************
      *
      *       TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

   /** CHARGER LE DATASET **/

    val parquetFileDF = spark.read.parquet("/Users/baptisteaubert/Desktop/Big_Data_Telecom_Paris/Semestre1/HadoopSpark/Spark")
    parquetFileDF.show()


    /** TF-IDF **/

    /* Stage 1  */
    val   tokenizer  =   new   RegexTokenizer()       .setPattern( "\\W+")
      .setGaps( true )
      .setInputCol( "text" )
      .setOutputCol( "tokens" )

    /* Stage 2  */
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("StopRemove")

    /* Stage 3  */
    val cvModel: CountVectorizer = new CountVectorizer()
      .setInputCol("StopRemove")
      .setOutputCol("Counting")

    /* Stage 4  */
    val idf = new IDF().setInputCol("Counting").setOutputCol("TFidf")

    /* Stage 5  */

    val indexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    /* Stage 6  */
    val indexer2 = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    /* Stage 7 */
    val assembler = new VectorAssembler()
      .setInputCols(Array("TFidf", "days_campaign", "hours_prepa", "goal", "country_indexed", "currency_indexed"))
      .setOutputCol("features")

    /* Stage 8 */

    val   lr  =   new   LogisticRegression()
      .setElasticNetParam( 0.0 )
      .setFitIntercept( true )
      .setFeaturesCol( "features" )
      .setLabelCol( "final_status" )
      .setStandardization( true )
      .setPredictionCol( "predictions" )
      .setRawPredictionCol( "raw_predictions" )
      .setThresholds( Array ( 0.7, 0.3))
      .setTol( 1.0e-6 )
      .setMaxIter(300)


    /* Pipeline */
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, cvModel, idf, indexer, indexer2, assembler, lr))

    val Array(training, test) = parquetFileDF.randomSplit(Array(0.9, 0.1), seed = 12345)

    //val model = pipeline.fit(training)

    val paramGrid = new ParamGridBuilder()
      .addGrid(cvModel.minDF, Array(20.0, 75.0,20))
      .addGrid(lr.regParam, Array(10e-8,10e-4,10e-2))
      .build()


    val f1score = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    val trainValidationSplit= new TrainValidationSplit()
      .setEstimator(lr)
      .setEstimator(pipeline)
      .setEvaluator(f1score)
      .setTrainRatio(0.70)
      .setEstimatorParamMaps(paramGrid)

    val model = trainValidationSplit.fit(training)
    val df_WithPredictions=model.transform(test)

    print("f1 evaluate : ", f1score.evaluate(df_WithPredictions))

    df_WithPredictions.groupBy("final_status", "predictions").count.show()

    /*Appliquer le meilleur modèle trouvé avec la grid-search aux données test. Mettre
les résultats dans le dataFrame df_WithPredictions. Afficher le f1-score du
modèle sur les données de test.
m. Afficher df_WithPredictions.groupBy("final_status", "predictions").count.show()/
*/

  }
}
