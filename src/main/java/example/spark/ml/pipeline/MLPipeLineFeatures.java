/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package example.spark.ml.pipeline;

import example.spark.ml.pipeline.utils.Utils;
import java.util.List;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * A feature transformer that merges multiple columns into a vector column.
 *
 * @author ranjeet
 */
public class MLPipeLineFeatures {

    private static Logger log = Logger.getLogger(MLPipeLineFeatures.class);

    public static void main(String[] args) {
        log.setLevel(Level.INFO);
        Utils utils = new Utils();
        SparkSession sparkSession = utils.getSparkSession(MLPipeLineFeatures.class.getSimpleName());
        log.info("New Spark Session started ");
        //Load and Clean data (clean means filter columns)
        Dataset<Row> salaryDF = utils.loadSalaryCsvTrain(sparkSession);

        //
        List<PipelineStage> pipelineStages = utils.buildFeaturesPipeLine();
        PipelineStage[] stages = pipelineStages.toArray(new PipelineStage[pipelineStages.size()]);
        Pipeline pipeline = new Pipeline().setStages(stages);

        //Model generate and data tronsform
        Dataset<Row> featurisedsalaryDF = pipeline.fit(salaryDF).transform(salaryDF);
        featurisedsalaryDF.select("features").show(false);
        log.info("**** Done ******");

    }
}
