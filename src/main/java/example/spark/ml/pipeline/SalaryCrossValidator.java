/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package example.spark.ml.pipeline;

import example.spark.ml.pipeline.utils.Utils;
import java.io.File;
import java.io.IOException;
import java.util.List;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.classification.BinaryLogisticRegressionSummary;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionTrainingSummary;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;

/**
 *
 * @author ranjeet
 */
public class SalaryCrossValidator {

    private static Logger log = Logger.getLogger(SalaryCrossValidator.class);

    public static void main(String[] args) {

        log.setLevel(Level.INFO);

        Utils utils = new Utils();

        SparkSession sparkSession = utils.getSparkSession(SalaryCrossValidator.class.getSimpleName());

        log.info("New Spark Session started ");
        //Load resource data
        Dataset<Row> salaryDF = utils.loadSalaryCsvTrain(sparkSession);
        Dataset<Row> testSalaryDf = utils.loadSalaryCsvTest(sparkSession);

        //build feature pipeline for dataset
        List<PipelineStage> pipelineStages = utils.buildFeaturesPipeLine();

        //Transform data to  ML columns like [0,1] on salary column | use for classification
        StringIndexer salaryIndexer = utils.buildStringIndexerFeature("salary", "label");

        pipelineStages.add(salaryIndexer);

        LogisticRegression logisticRegression = new LogisticRegression();
        //Add LogisticRegression to pipeline
        pipelineStages.add(logisticRegression);

        PipelineStage[] stages = pipelineStages.toArray(new PipelineStage[pipelineStages.size()]);

        Pipeline pipelineWithLogisticRegression = new Pipeline().setStages(stages);

        //traning training data
//        Dataset<Row> training = pipelineWithLogisticRegression.fit(salaryDF).transform(salaryDF);
//        training.select("features", "label").show(false);
        ParamMap[] paramMaps = new ParamGridBuilder()
                .addGrid(logisticRegression.regParam(), new double[]{0.1, 0.01})
                .addGrid(logisticRegression.maxIter(), new int[]{20, 30})
                .build();

        BinaryClassificationEvaluator evaluator = new BinaryClassificationEvaluator();

        CrossValidator crossValidator = new CrossValidator()
                .setEstimator(pipelineWithLogisticRegression)
                .setNumFolds(5)
                .setEstimatorParamMaps(paramMaps)
                .setEvaluator(evaluator);

        //clean, transform salary data using pipeline and pass to logistic model
        CrossValidatorModel crossValidatorModel = crossValidator.fit(salaryDF);

        Dataset<Row> predictions = crossValidatorModel.transform(testSalaryDf);

        ParamMap areaUnderROCparamMap = new ParamMap();
        areaUnderROCparamMap.put(evaluator.metricName(), "areaUnderROC");

        double areaUnderROCEvaluate = evaluator.evaluate(predictions, areaUnderROCparamMap);
        log.info("areaUnderROCEvaluate :" + areaUnderROCEvaluate);

        PipelineModel bestModel = (PipelineModel) crossValidatorModel.bestModel();
        Transformer[] bestModelstages = bestModel.stages();
        log.info("No. of bestModelstages : " + bestModelstages.length);

        LogisticRegressionModel logisticRegressionModel = (LogisticRegressionModel) bestModelstages[bestModelstages.length - 1];

        log.info("bestModelstages regParam = " + logisticRegressionModel.getRegParam());

        LogisticRegressionTrainingSummary trainingSummary = logisticRegressionModel.summary();
        // Obtain the loss per iteration.
        double[] objectiveHistory = trainingSummary.objectiveHistory();
        for (double lossPerIteration : objectiveHistory) {
            System.out.println(lossPerIteration);
        }

        // Obtain the metrics useful to judge performance on test data.
        // We cast the summary to a BinaryLogisticRegressionSummary since the problem is a binary
        // classification problem.
        BinaryLogisticRegressionSummary binarySummary
                = (BinaryLogisticRegressionSummary) trainingSummary;

        // Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
        Dataset<Row> roc = binarySummary.roc();
        log.info("******************************************************");
        log.info("receiver-operating characteristic (roc)");
        roc.show();
        roc.select("FPR").show();
        log.info("********* areaUnderROC **********\n" + binarySummary.areaUnderROC());

        // Get the threshold corresponding to the maximum F-Measure and rerun LogisticRegression with
        // this selected threshold.
        Dataset<Row> fMeasure = binarySummary.fMeasureByThreshold();
        double maxFMeasure = fMeasure.select(functions.max("F-Measure")).head().getDouble(0);
        double bestThreshold = fMeasure.where(fMeasure.col("F-Measure").equalTo(maxFMeasure))
                .select("threshold").head().getDouble(0);
        logisticRegressionModel.setThreshold(bestThreshold);

        try {
            String saveModel = "/tmp/logisticRegressionModel_best";
            File saveModelFile = new File(saveModel);
            if (saveModelFile.exists()) {
                saveModelFile.delete();
            }
            logisticRegressionModel.save(saveModel);
        } catch (IOException ex) {
            log.error("Error while saving model ", ex);
        }
        log.info("**** Done ******");

    }
}
