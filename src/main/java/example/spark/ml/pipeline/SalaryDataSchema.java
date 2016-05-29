package example.spark.ml.pipeline;

import example.spark.ml.pipeline.utils.Utils;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class SalaryDataSchema {

    private static Logger log = Logger.getLogger(SalaryDataSchema.class);

    public static void main(String[] args) {
        log.setLevel(Level.INFO);
        Utils utils = new Utils();
        SparkSession sparkSession = utils.getSparkSession(SalaryDataSchema.class.getSimpleName());
        log.info("Spark Session started " + sparkSession.toString());
        //Load and Clean data (clean means filter columns)
        Dataset<Row> salaryDataset = utils.loadSalaryCsvTrain(sparkSession);
        log.info("Finally print clean schema");
        salaryDataset.printSchema();
        log.info("**** Done ******");
    }

}
