package examples;

import cf4j.Kernel;
import cf4j.Processor;
import cf4j.model.matrixFactorization.Bmf;
import cf4j.model.matrixFactorization.Ndcgumf;
import cf4j.model.matrixFactorization.Pmf;
import cf4j.model.predictions.FactorizationPrediction;
import cf4j.qualityMeasures.MAE;
import cf4j.qualityMeasures.Precision;

/**
 * Compare MAE and Precision of PMF and nDCGUMF.
 * @author Eduardo Cobos
 */

public class Example4 {

    // --- PARAMETERS DEFINITION ------------------------------------------------------------------

    private static String dataset = "C:\\Users\\Eduardo\\Documents\\Trabajo\\Publicaciones\\Fernando\\ml-1m\\ratings.dat";
    private static double testItems = 0.2; // 20% test items
    private static double testUsers = 0.2; // 20% test users

    private static int numRecommendations = 10;
    private static double threshold = 4.0;

    private static int pmf_numTopics = 15;
    private static int pmf_numIters = 50;
    private static double pmf_lambda = 0.055;

    private static int ndcgumf_numTopics = 6;
    private static int ndcgumf_numIters = 50;
    //private static double ndcgumf_alpha = 0.8;
    //private static double ndcgumf_beta = 5;

    // --------------------------------------------------------------------------------------------

    public static void main (String [] args) {

        // Load the database
        Kernel.getInstance().open(dataset, testUsers, testItems, "::");


        // PMF
        Pmf pmf = new Pmf (pmf_numTopics, pmf_numIters, pmf_lambda);
        pmf.train();

        Processor.getInstance().testUsersProcess(new FactorizationPrediction(pmf));

        System.out.println("\nPMF:");

        Processor.getInstance().testUsersProcess(new MAE());
        System.out.println("- MAE: " + Kernel.gi().getQualityMeasure("MAE"));

        Processor.getInstance().testUsersProcess(new Precision(numRecommendations, threshold));
        System.out.println("- Precision: " + Kernel.gi().getQualityMeasure("Precision"));


        // Ndcgumf
        Ndcgumf ndcgumf = new Ndcgumf (ndcgumf_numTopics, ndcgumf_numIters);
        ndcgumf.train();

        Processor.getInstance().testUsersProcess(new FactorizationPrediction(ndcgumf));

        System.out.println("\nNdcgumf:");

        Processor.getInstance().testUsersProcess(new MAE());
        System.out.println("- MAE: " + Kernel.gi().getQualityMeasure("MAE"));

        Processor.getInstance().testUsersProcess(new Precision(numRecommendations, threshold));
        System.out.println("- Precision: " + Kernel.gi().getQualityMeasure("Precision"));
    }
}
