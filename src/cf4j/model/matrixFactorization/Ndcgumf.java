package cf4j.model.matrixFactorization;

import cf4j.Item;
import cf4j.ItemsPartible;
import cf4j.Kernel;
import cf4j.Processor;
import cf4j.User;
import cf4j.UsersPartible;
import cf4j.utils.Methods;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

/**
 * Implements Probabilist Matrix Factorization: Ortega, F.  	&amp; Cobos, C.E. (2019).
 *
 * @author Eduardo Cobos
 */

public class Ndcgumf  implements FactorizationModel {

    private final static String USER_BIAS_KEY = "ndcgumf-user-bias";
    private final static String USER_FACTORS_KEY = "ndcgumf-user-factors";

    private final static String ITEM_BIAS_KEY = "ndcgumf-item-bias";
    private final static String ITEM_FACTORS_KEY = "ndcgumf-item-factors";

    private final static double DEFAULT_BETA = 2;
    private final static double DEFAULT_GAMMA = 0.01;
    private final static double DEFAULT_LAMBDA = 0.1;

    /**
     * Learning rate: 11 by default
     */
    private double beta;

    /**
     * Learning rate: 0.01 by default
     */
    private double gamma;

    /**
     * Regularization parameter: 0.1 by default
     */
    private double lambda;

    /**
     * Number of latent factors
     */
    private int numFactors;

    /**
     * Number of iterations
     */
    private int numIters;

    /**
     * Enable biases
     */
    private boolean biases;

    /**
     * Model constructor
     * @param numFactors Number of factors
     * @param numIters Number of iterations
     */
    public Ndcgumf (int numFactors, int numIters)	{
        this(numFactors, numIters, DEFAULT_LAMBDA, DEFAULT_GAMMA, true);
    }

    /**
     * Model constructor
     * @param numFactors Number of factors
     * @param numIters Number of iterations
     * @param lambda Regularization parameter
     */
    public Ndcgumf (int numFactors, int numIters, double lambda) {
        this(numFactors, numIters, lambda, DEFAULT_GAMMA, true);
    }

    /**
     * Model constructor
     * @param numFactors Number of factors
     * @param numIters Number of iterations
     * @param lambda Regularization parameter
     * @param gamma Learning rate parameter
     */
    public Ndcgumf (int numFactors, int numIters, double lambda, double gamma) {
        this(numFactors, numIters, lambda, gamma, true);
    }

    /**
     * Model constructor
     * @param numFactors Number of factors
     * @param numIters Number of iterations
     * @param lambda Regularization parameter
     * @param biases Enable/disable biases in the model
     */
    public Ndcgumf (int numFactors, int numIters, double lambda, boolean biases) {
        this(numFactors, numIters, lambda, DEFAULT_GAMMA, biases);
    }

    /**
     * Model constructor
     * @param numFactors Number of factors
     * @param numIters Number of iterations
     * @param lambda Regularization parameter
     * @param gamma Learning rate parameter
     * @param biases Enable/disable biases in the model
     */
    public Ndcgumf (int numFactors, int numIters, double lambda, double gamma, boolean biases) {

        this.numFactors = numFactors;
        this.numIters = numIters;
        this.lambda = lambda;
        this.gamma = gamma;
        this.biases = biases;

        // Users initialization
        for (int u = 0; u < Kernel.gi().getNumberOfUsers(); u++) {
            this.setUserFactors(u, this.random(this.numFactors, -1, 1));
        }

        // Items initialization
        for (int i = 0; i < Kernel.gi().getNumberOfItems(); i++) {
            this.setItemFactors(i, this.random(this.numFactors, -1, 1));
        }

        // Initialize bias if needed
        if (this.biases) {

            // Users bias initialization
            for (int u = 0; u < Kernel.gi().getNumberOfUsers(); u++) {
                this.setUserBias(u, this.random(-1, 1));
            }

            // Items bias initialization
            for (int i = 0; i < Kernel.gi().getNumberOfItems(); i++) {
                this.setItemBias(i, this.random(-1, 1));
            }
        }
    }

    /**
     * Get the number of topics of the model
     * @return Number of topics
     */
    public int getNumberOfTopics () {
        return this.numFactors;
    }

    /**
     * Get the regularization parameter of the model
     * @return Lambda
     */
    public double getLambda () {
        return this.lambda;
    }

    /**
     * Get the learning rate parameter of the model
     * @return Gamma
     */
    public double getGamma () {
        return this.gamma;
    }

    /**
     * Estimate the latent model factors
     */
    public void train () {

        System.out.println("\nProcessing NDCGUMF...");

        for (int iter = 1; iter <= this.numIters; iter++) {

            // ALS: fix q_i and update p_u -> fix p_u and update q_i
            System.out.println("Update Item...");
            Processor.getInstance().itemsProcess(new Ndcgumf.UpdateItemsFactors(), false);
            System.out.println("Update User...");
            Processor.getInstance().usersProcess(new Ndcgumf.UpdateUsersFactors(), false);
            System.out.println("Update ItemCopy...");
            Processor.getInstance().itemsProcess(new Ndcgumf.UpdateItemsCopyFactors(), false);

            if ((iter % 10) == 0) System.out.print(".");
            if ((iter % 100) == 0) System.out.println(iter + " iterations");
        }
    }

    /**
     * Get user factors
     * @param userIndex User index
     * @return User factors
     */
    public double [] getUserFactors (int userIndex) {
        User user = Kernel.gi().getUsers()[userIndex];
        return (double []) user.get(USER_FACTORS_KEY);
    }

    /**
     * Set user factors
     * @param userIndex User index
     * @param factors User factors
     */
    private void setUserFactors (int userIndex, double [] factors) {
        User user = Kernel.gi().getUsers()[userIndex];
        user.put(USER_FACTORS_KEY, factors);
    }

    /**
     * Get item factors
     * @param itemIndex Item index
     * @return Item factors
     */
    public double [] getItemFactors (int itemIndex) {
        Item item = Kernel.gi().getItems()[itemIndex];
        return (double []) item.get(ITEM_FACTORS_KEY);
    }

    /**
     * Get item copy factors
     * @param itemIndex Item copy index
     * @return Item copy factors
     */
    public double [] getItemCopyFactors (int itemIndex) {
        Item item = Kernel.gi().getItemsCopy()[itemIndex];
        return (double []) item.get(ITEM_FACTORS_KEY);
    }

    /**
     * Set item factors
     * @param itemIndex Item index
     * @param factors Item factors
     */
    private void setItemFactors (int itemIndex, double [] factors) {
        Item item = Kernel.gi().getItems()[itemIndex];
        item.put(ITEM_FACTORS_KEY, factors);
    }

    /**
     * Set item copy factors
     * @param itemIndex Item copy index
     * @param factors Item copy factors
     */
    private void setItemCopyFactors (int itemIndex, double [] factors) {
        Item item = Kernel.gi().getItemsCopy()[itemIndex];
        item.put(ITEM_FACTORS_KEY, factors);
    }

    /**
     * Get user bias (if enabled)
     * @param userIndex User index
     * @return User bias or null
     */
    public double getUserBias (int userIndex) {
        User user = Kernel.gi().getUsers()[userIndex];
        return (Double) user.get(USER_BIAS_KEY);
    }

    /**
     * Set user bias
     * @param userIndex User index
     * @param bias User bias
     */
    private void setUserBias (int userIndex, double bias) 	{
        User user = Kernel.gi().getUsers()[userIndex];
        user.put(USER_BIAS_KEY, bias);
    }

    /**
     * Get item bias (if needed)
     * @param itemIndex Item index
     * @return Item bias
     */
    public double getItemBias (int itemIndex) {
        Item item = Kernel.gi().getItems()[itemIndex];
        return (Double) item.get(ITEM_BIAS_KEY);
    }

    /**
     * Set item bias
     * @param itemIndex Item index
     * @param bias Item bias
     */
    private void setItemBias (int itemIndex, double bias) {
        Item item = Kernel.gi().getItems()[itemIndex];
        item.put(ITEM_BIAS_KEY, bias);
    }

    /**
     * Computes a rating prediction
     * @param userIndex User index
     * @param itemIndex Item index
     * @return Prediction
     */
    public double getPrediction (int userIndex, int itemIndex) {
        double [] factors_u = this.getUserFactors(userIndex);
        double [] factors_i = this.getItemCopyFactors(itemIndex);

        if (this.biases) {
            double average = Kernel.gi().getRatingAverage();

            double bias_u = this.getUserBias(userIndex);
            double bias_i = this.getItemBias(itemIndex);

            return average + bias_u + bias_i + Methods.dotProduct(factors_u, factors_i);
        }
        else {
            return Methods.dotProduct(factors_u, factors_i);
        }
    }

    /**
     * Auxiliary inner class to parallelize user factors computation
     * @author Fernando Ortega
     */
    private class UpdateUsersFactors implements UsersPartible {

        @Override
        public void beforeRun() { }

        @Override
        public void run (int userIndex) {

            User user = Kernel.gi().getUsers()[userIndex];

            int itemIndex = 0;
            double idcgu;
            double smi, smiprima, gradiente;

            double [] q_iCopy;
            double [] q_jCopy;

            double [] p_u = Ndcgumf.this.getUserFactors(userIndex);
            double [] p_u_aux = new double[p_u.length];

            System.out.println(userIndex);

            for (int j = 0; j < user.getNumberOfRatings(); j++) {

                while (Kernel.gi().getItems()[itemIndex].getItemCode() < user.getItems()[j]) itemIndex++;

                smi = Ndcgumf.this.softmax(userIndex,itemIndex);
                idcgu = Ndcgumf.this.getIdcgu(userIndex);
                // Get gradiente

                gradiente = 1/idcgu*
                        (Math.pow(2,user.getRatings()[j])-1)/
                        (Math.pow(Math.log(Ndcgumf.this.getPos(userIndex,itemIndex) + 1),3))*
                        smi;

                // Update p_u
                q_iCopy = Ndcgumf.this.getItemCopyFactors(itemIndex);

                for (int k = 0; k < Ndcgumf.this.numFactors; k++)	{

                    int itemJndexPrima = 0;
                    smiprima = 0;

                    for (int w = 0; w < user.getNumberOfRatings(); w++) {

                        while (Kernel.gi().getItems()[itemIndex].getItemCode() < user.getItems()[j]) itemJndexPrima++;

                        q_jCopy = Ndcgumf.this.getItemCopyFactors(itemJndexPrima);

                        smiprima += q_jCopy[k] * Ndcgumf.this.softmax(userIndex,itemJndexPrima);;
                    }
                    p_u_aux[k] -= Ndcgumf.this.gamma * Ndcgumf.this.beta * (user.getNumberOfRatings()-1) *
                            Math.log(2) * gradiente * (q_iCopy[k]-smiprima);
                }

                //Ndcgumf.this.setUserFactors(userIndex, p_u);

                // Update biases if needed
                /*if (Ndcgumf.this.biases) {
                    double b_u = Ndcgumf.this.getUserBias(userIndex);

                    b_u += Ndcgumf.this.gamma * (error - Ndcgumf.this.lambda * b_u);

                    Ndcgumf.this.setUserBias(userIndex, b_u);
                }*/
            }
            System.out.println(userIndex);
            for (int k = 0; k < Ndcgumf.this.numFactors; k++) {
                p_u[k] -= p_u_aux[k] + Ndcgumf.this.gamma * Ndcgumf.this.lambda * p_u[k];
            }
        }

        @Override
        public void afterRun() { }
    }

    /**
     * Auxiliary inner class to parallelize item factors computation
     * @author Fernando Ortega
     */
    private class UpdateItemsFactors implements ItemsPartible {

        @Override
        public void beforeRun() { }

        @Override
        public void afterRun() { }

        @Override
        public void run(int itemIndex) {

            System.out.println(itemIndex);

            double idcgu;
            double smi, smiprima;

            double [] q_i = Ndcgumf.this.getItemFactors(itemIndex);
            double [] q_iCopy = Ndcgumf.this.getItemCopyFactors(itemIndex);

            Item item = Kernel.gi().getItems()[itemIndex];

            int userIndex = 0;

            for (int v = 0; v < item.getNumberOfRatings(); v++)
            {
                User user = Kernel.gi().getUsers()[userIndex];

                double gradiente = 0;

                if (itemIndex==0){
                    System.out.println("" + itemIndex + "While start" + v);
                }

                while (user.getUserCode() < item.getUsers()[v]){

                    int itemIndexPrima = 0;

                    for(int w = 0; w < user.getNumberOfRatings(); w++)
                    {
                        while(Kernel.gi().getItems()[itemIndexPrima].getItemCode() < user.getItems()[w]) itemIndexPrima++;

                        if(itemIndex != itemIndexPrima){
                            idcgu = Ndcgumf.this.getIdcgu(userIndex);
                            smi = Ndcgumf.this.softmax(userIndex,itemIndex);
                            smiprima =Ndcgumf.this.softmax(userIndex,itemIndexPrima);

                            gradiente -= 1/idcgu*
                                (Math.pow(2,user.getRatings()[w])-1)/
                                (Math.pow(Math.log(Ndcgumf.this.getPos(userIndex,itemIndexPrima) + 1),3))*
                                smi*smiprima;
                        }

                    }

                    userIndex++;

                    user = Kernel.gi().getUsers()[userIndex];
                }
                if (itemIndex==0){
                    System.out.println("" + itemIndex + "While fi");
                }

                int itemIndexPrima = 0;

                if (itemIndex==0){
                    System.out.println("" + itemIndex + "For start");
                }

                for(int w = 0; w < user.getNumberOfRatings(); w++)
                {
                    while(Kernel.gi().getItems()[itemIndexPrima].getItemCode() < user.getItems()[w]) itemIndexPrima++;

                    if(itemIndex != itemIndexPrima){
                        idcgu = Ndcgumf.this.getIdcgu(userIndex);
                        smi = Ndcgumf.this.softmax(userIndex,itemIndex);
                        smiprima =Ndcgumf.this.softmax(userIndex,itemIndexPrima);

                        gradiente -= 1/idcgu*
                                (Math.pow(2,user.getRatings()[w])-1)/
                                (Math.pow(Math.log(Ndcgumf.this.getPos(userIndex,itemIndexPrima) + 1),3))*
                                smi*smiprima;
                    }

                }

                if (itemIndex==0){
                    System.out.println("" + itemIndex + "For fi");
                }

                smi = Ndcgumf.this.softmax(userIndex,itemIndex);
                idcgu = Ndcgumf.this.getIdcgu(userIndex);
                // Get gradiente
                gradiente += 1/idcgu*
                        (Math.pow(2,item.getRatings()[v])-1)/
                        (Math.pow(Math.log(Ndcgumf.this.getPos(userIndex,itemIndex) + 1),3))*
                        smi*(1-smi);

                // Update p_u
                double [] p_u = Ndcgumf.this.getUserFactors(userIndex);

                if (itemIndex==0){
                    System.out.println("" + itemIndex + "For 2 start");
                }

                for (int k = 0; k < Ndcgumf.this.numFactors; k++) {
                    q_i[k] -= Ndcgumf.this.gamma * Ndcgumf.this.beta * (user.getNumberOfRatings()-1) *
                            Math.log(2) * p_u[k] * gradiente;
                }

                if (itemIndex==0){
                    System.out.println("" + itemIndex + "For 2 fi");
                }
            }

            System.out.println(itemIndex);

            for (int k = 0; k < Ndcgumf.this.numFactors; k++) {
                q_i[k] -= Ndcgumf.this.gamma * Ndcgumf.this.lambda * q_iCopy[k];
            }
        }
    }

    /**
     * Auxiliary inner class to parallelize item factors computation
     * @author Fernando Ortega
     */
    private class UpdateItemsCopyFactors implements ItemsPartible {

        @Override
        public void beforeRun() { }

        @Override
        public void afterRun() { }

        @Override
        public void run(int itemIndex) {

            double [] q_i = Ndcgumf.this.getItemFactors(itemIndex);
            double [] q_iCopy = Ndcgumf.this.getItemCopyFactors(itemIndex);

            q_iCopy = q_i;
        }
    }


    /**
     * Get a random number between min and max
     * @param min Minimum random value
     * @param max Maximum random value
     * @return Random value between min and max
     */
    private double random (double min, double max) {
        return Math.random() * (max - min) + min;
    }

    /**
     * Get an array of random numbers
     * @param size Array length
     * @param min Minimum random value
     * @param max Maximum random value
     * @return Array of randoms
     */
    private double [] random (int size, double min, double max) {
        double [] d = new double [size];
        for (int i = 0; i < size; i++) d[i] = this.random(min, max);
        return d;
    }

    private double getIdcgu(int userIndex){

        double idcgu = 0;

        User user = Kernel.gi().getUsers()[userIndex];

        ArrayList<Double> valores = new ArrayList<>();

        for (int j = 0; j < user.getNumberOfRatings(); j++) {

            //Sacar lista de valores para ordenarlos
            valores.add(user.getRatings()[j]);
        }

        //Ordenar y calcular idcgu
        Comparator<Double> comparador = Collections.reverseOrder();
        Collections.sort(valores, comparador);

        for(int i = 0; i<valores.size();i++){

            idcgu += (Math.log(2)*(Math.pow(2,valores.get(i))-1))/(Math.log(i+2));

        }

        return idcgu;

    }

    private double getPos(int userIndex, int itemIndex){

        User user = Kernel.gi().getUsers()[userIndex];

        double pos = user.getNumberOfRatings() - (user.getNumberOfRatings()-1)*this.softmax(userIndex, itemIndex);

        return pos;

    }

    private double softmax(int userIndex, int itemIndex){

        double softmax = 0;

        User user = Kernel.gi().getUsers()[userIndex];

        int itemJndex = 0;

        for (int j = 0; j < user.getNumberOfRatings(); j++) {

            while (Kernel.gi().getItems()[itemJndex].getItemCode() < user.getItems()[j]) itemJndex++;

            softmax += Math.exp(this.beta * Ndcgumf.this.getPrediction(userIndex, itemJndex));
        }

        softmax = Math.exp(this.beta * Ndcgumf.this.getPrediction(userIndex, itemIndex)) / softmax;

        return softmax;

    }
}
