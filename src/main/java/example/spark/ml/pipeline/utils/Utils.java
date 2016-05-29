/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package example.spark.ml.pipeline.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.apache.log4j.Logger;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SparkSession;

public class Utils {

    private static Logger log = Logger.getLogger(Utils.class);
    private String salary_csv = "adult.data";
    private String test_salary_csv = "adult.test";

    /**
     * *
     * SparkSession is the new entry point to programming Spark with the Dataset
     * and DataFrame API’s .It encompasses SQLContext, SparkContext etc.,
     *
     * @param sessionName
     * @return SparkSession
     */
    public SparkSession getSparkSession(String sessionName) {
        SparkSession sparkSession = SparkSession
                .builder()
                .master("local[2]")
                .appName(sessionName)
                .getOrCreate();
        log.info("SparkSession Created :" + sessionName);
        return sparkSession;

    }

    public Dataset<Row> loadSalaryCsvTrain(SparkSession sparkSession) {

        log.info("Get Dataset<Row> from resource file : " + this.salary_csv);
        String salaryCsv = Utils.class.getClassLoader().getResource(this.salary_csv).getPath();
        return this.getDatasetRow(salaryCsv, sparkSession);
    }

    public Dataset<Row> loadSalaryCsvTest(SparkSession sparkSession) {

        log.info("Get Dataset<Row> from resource file : " + this.test_salary_csv);
        String salaryCsv = Utils.class.getClassLoader().getResource(this.test_salary_csv).getPath();
        return this.getDatasetRow(salaryCsv, sparkSession);
    }

    private Dataset<Row> cleanSalaryCsv(String resourceFile, SparkSession sparkSession) {
        Dataset<Row> df = this.getDatasetRow(resourceFile, sparkSession);

        Dataset<Row> cleanDataset = df.select("workclass", "occupation", "native_country");
        log.info("Selected columns :workclass occupation native_country ");
        return cleanDataset;

    }

    public Dataset<Row> getDatasetRow(String filename, SparkSession sparkSession) {
        log.info("Get Dataset<Row> from resource file : " + filename);
        SQLContext sqlContext = new SQLContext(sparkSession);
        Dataset<Row> df = sqlContext.read()
                .format("com.databricks.spark.csv")
                .option("inferSchema", "true")
                .option("header", "true")
                .load(filename);
        //Prints the schema of the underlying Dataset to the console in a nice tree format.
        log.info("Actual schema :");
        df.printSchema();
        return df;
    }

    /**
     * *
     * A label indexer that maps a string column of labels to an ML column of
     * label indices. If the input column is numeric, we cast it to string and
     * index the string values. The indices are in [0, numLabels), ordered by
     * label frequencies. So the most frequent label gets index 0.
     *
     * @param outputColn name label
     * @param inputColn name , use for labeling
     * @param Dataset<Row>
     * @return transformed Dataset<Row>
     */
    public Dataset<Row> getStringIndexer(Dataset<Row> dataset, String inputColn, String labelColumnNameOut) {
        StringIndexer labelIndexer = this.buildStringIndexerFeature(inputColn, labelColumnNameOut);
        StringIndexerModel labelIndexerModel = labelIndexer.fit(dataset);
        log.info("labels are : " + Arrays.toString(labelIndexerModel.labels()));
        Dataset<Row> transformDataset = labelIndexerModel.transform(dataset);
        return transformDataset;

    }

    /**
     * *
     * Build Feature pipeline for string type column
     *
     * @param colName
     * @return StringIndexer, OneHotEncoder PipelineStages
     */
    private List<PipelineStage> buildFeaturesPipelineStage(String colName) {
        StringIndexer stringIndexer = this.buildStringIndexerFeature(colName);
        //StringIndexer output put in OneHotEncoder
        OneHotEncoder oneHotEncoder = this.buildOneHotEncoderFeature(stringIndexer.getOutputCol());
        //Pass OneHotEncoder output to Vector Assembler
        VectorAssembler vectorAssembler = this.buildVectorAssemblerFeature(oneHotEncoder.getOutputCol());
        PipelineStage[] PipelineStages = {stringIndexer, oneHotEncoder};
        return Arrays.<PipelineStage>asList(PipelineStages);
    }

    private StringIndexer buildStringIndexerFeature(String colName) {
        StringIndexer stringIndexer = new StringIndexer()
                .setInputCol(colName)
                .setOutputCol(colName + "_index");

        return stringIndexer;
    }

    public StringIndexer buildStringIndexerFeature(String colName, String labelColumnNameOut) {
        StringIndexer stringIndexer = new StringIndexer()
                .setInputCol(colName)
                .setOutputCol(labelColumnNameOut);

        return stringIndexer;
    }

    /**
     * *
     * A one-hot encoder that maps a column of category indices to a column of
     * binary vectors, with at most a single one-value per row that indicates
     * the input category index. For example with 5 categories, an input value
     * of 2.0 would map to an output vector of [0.0, 0.0, 1.0, 0.0]. The last
     * category is not included by default (configurable via
     * OneHotEncoder!.dropLast because it makes the vector entries sum up to
     * one, and hence linearly dependent. So an input value of 4.0 maps to [0.0,
     * 0.0, 0.0, 0.0]. Note that this is different from scikit-learn's
     * OneHotEncoder, which keeps all categories. The output vectors are sparse.
     *
     * @param colName
     * @return
     */
    private OneHotEncoder buildOneHotEncoderFeature(String colName) {
        OneHotEncoder oneHotEncoder = new OneHotEncoder()
                .setInputCol(colName)
                .setOutputCol(colName + "_onehotindex");
        return oneHotEncoder;

    }

    /**
     * *
     * A feature transformer that merges multiple columns into a vector column.
     *
     * @param colName
     * @return VectorAssembler
     */
    private VectorAssembler buildVectorAssemblerFeature(String colName) {
        VectorAssembler vectorAssembler = new VectorAssembler()
                .setOutputCol(colName)
                .setOutputCol("features");
        return vectorAssembler;
    }

    /**
     * *
     * A feature transformer that merges multiple columns into a vector column.
     *
     * @param colns
     * @return VectorAssembler
     */
    private VectorAssembler buildVectorAssemblerFeature(String[] colns) {
        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(colns)
                .setOutputCol("features");
        return vectorAssembler;
    }

    public List<PipelineStage> buildFeaturesPipeLine() {
        List<PipelineStage> pipelineStages = new ArrayList<>();
        List<PipelineStage> workclassStages = this.buildFeaturesPipelineStage("workclass");
        List<PipelineStage> educationStages = this.buildFeaturesPipelineStage("education");
        List<PipelineStage> occupationStages = this.buildFeaturesPipelineStage("occupation");
        List<PipelineStage> marital_statusStages = this.buildFeaturesPipelineStage("marital_status");
        List<PipelineStage> relationshipStages = this.buildFeaturesPipelineStage("relationship");
        List<PipelineStage> sexStages = this.buildFeaturesPipelineStage("sex");
        List<PipelineStage> native_countryStages = this.buildFeaturesPipelineStage("native_country");
        List<PipelineStage> raceStages = this.buildFeaturesPipelineStage("race");

        pipelineStages.addAll(workclassStages);
        pipelineStages.addAll(educationStages);
        pipelineStages.addAll(occupationStages);
        pipelineStages.addAll(marital_statusStages);
        pipelineStages.addAll(relationshipStages);
        pipelineStages.addAll(sexStages);
        pipelineStages.addAll(native_countryStages);
        pipelineStages.addAll(raceStages);

        String[] colns = {"workclass_index_onehotindex", "occupation_index_onehotindex", "relationship_index_onehotindex",
            "marital_status_index_onehotindex", "sex_index_onehotindex", "age", "education_num", "hours_per_week"};
        VectorAssembler numDataVectorAssembler = this.buildVectorAssemblerFeature(colns);
        pipelineStages.add(numDataVectorAssembler);

        return pipelineStages;

    }

}
