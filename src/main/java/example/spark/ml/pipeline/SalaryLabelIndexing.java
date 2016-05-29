/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package example.spark.ml.pipeline;

import example.spark.ml.pipeline.utils.Utils;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 *
 * @author ranjeet
 */
public class SalaryLabelIndexing {

    private static Logger log = Logger.getLogger(SalaryLabelIndexing.class);

    public static void main(String[] args) {
        log.setLevel(Level.INFO);
        Utils utils = new Utils();
        SparkSession sparkSession = utils.getSparkSession(SalaryLabelIndexing.class.getSimpleName());
        log.info("Spark Session started " + sparkSession.toString());
        //Load salary resource data
        Dataset<Row> salaryDataset = utils.loadSalaryCsvTrain(sparkSession);

        //Transform data to  ML columns like [0,1] on salary column
        String labelColumnName = "salary";
        String labelColumnNameOut = "salary_label";
        Dataset<Row> transformLabelIndexer = utils.getStringIndexer(salaryDataset, labelColumnName, labelColumnNameOut);
        transformLabelIndexer.select(labelColumnName, labelColumnNameOut).show(20);
        log.info("**** Done ******");
    }
}
