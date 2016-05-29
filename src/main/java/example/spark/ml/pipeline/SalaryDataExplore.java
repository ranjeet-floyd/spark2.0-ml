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
public class SalaryDataExplore {

    private static Logger log = Logger.getLogger(SalaryDataExplore.class);

    public static void main(String[] args) {
        log.setLevel(Level.INFO);
        Utils utils = new Utils();
        SparkSession sparkSession = utils.getSparkSession(SalaryDataExplore.class.getSimpleName());
        log.info("Spark Session started " + sparkSession.toString());
        //Load and Clean data (clean means filter columns)
        Dataset<Row> salaryDataset = utils.loadSalaryCsvTrain(sparkSession);
        long salaryDatasetCount = salaryDataset.count();
        log.info("Total no of data: " + Long.toString(salaryDatasetCount));
        int numRows = 50;
        log.info("Show data for " + Integer.toString(numRows));
        salaryDataset.show(numRows);

        //Show some common info like count, mean, stddev, min, max for listed columns
        salaryDataset.describe("age", "education_num", "hours_per_week").show();

        //Computes a pair-wise frequency table of the given columns. |similar to confusion matrix
        salaryDataset.stat().crosstab("sex", "salary").show();

        //Returns a GroupedDataset where the data is grouped by the given Column expressions.
        //We have salary column  data as eitheir >50K or <=50K
        Dataset<Row> salaryGroup50K = salaryDataset.groupBy("salary").count();
        //Show group by count
        salaryGroup50K.show();
        log.info("**** Done ******");

    }

}
