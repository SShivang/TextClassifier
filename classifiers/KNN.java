package ir.classifiers;

import java.util.*;

import ir.vsr.*;

import ir.utilities.*;

/**
 * Abstract class specifying the functionality of a classifier. Provides methods for
 * training and testing a classifier
 *
 * @author Sugato Basu and Yuk Wah Wong
 */

public class KNN extends Classifier {

  /**
   * Trains the classifier on the training examples
   *
   * @param trainingExamples a list of Example objects that will be used
   *                         for training the classifier
   */

  // inverted index object for the training examples
  private InvertedIndex index = null;

  // flag to use the modified rocchio
  private int k;

  // name of the classifier
  public static String name = "KNN";

  // debug flag
  private boolean debug;

  // this is the training set in the form a list
  private HashMap<Example, HashMapVector> trainingSet;

  /*
    Constructor for the KNN class.
  */
  public KNN(String[] categories, boolean debug, int k) {
    this.categories = categories;
    this.debug = debug;
    this.k = k;
    trainingSet = new HashMap<Example, HashMapVector>();
    this.name += k;
  }

  /* This method returns the name of the classifier */
  public String getName(){
    return name;
  }

  /*
    This method is designed to train the KNN algorithm. It takes
    in a set of training examples. It sets the tf-idf of the training
    samples. If the tf-idf has already been calculated for the example
    it doesn't recalculate for the example.
  */
  public void train(List<Example> trainingExamples){

    // make an inverted index object to get the idfs
    index = new InvertedIndex(trainingExamples);

    // Initialize a training set
    trainingSet = new HashMap<Example, HashMapVector>();

    // for each example add to the trainingSet
    for (Example  e : trainingExamples){
      trainingSet.put(e, tfIdfVector(e.hashVector, e));
    }

  }

  /*
    This method calculates the tf idf vector for a given sample
    in place for space efficiency.
  */

  public HashMapVector tfIdfVector(HashMapVector original, Example e){

    // copy the original vector
    HashMapVector vector = original.copy();

    // load the map of idf
    Map<String, TokenInfo> map = index.tokenHash;

    // get the max weight
    double maxWeight = vector.maxWeight();

    // for each entry
    for (Map.Entry<String, Weight> entry : vector.hashMap.entrySet()) {

      // weight of the entry
      Weight weight = entry.getValue();

      // get the token
      String token = entry.getKey();

      // set the weight to the idf

      if (map.get(token) != null && maxWeight != 0.0){
        weight.setValue(map.get(token).idf * weight.getValue()/ maxWeight);
      }else{
        // the value of idf doesn't matter since the prototype will not contain it
        weight.setValue(0.0);
      }

    }

    return vector;

  }

  /**
   * Returns true if the predicted category of the test example matches the correct category,
   * false otherwise
   */
  public boolean test(Example testExample){

    HashMapVector tdIDFTestExample = tfIdfVector(testExample.hashVector, testExample);
    // keep track of all the testing in a testing set, this is a descending TreeMap
    TreeMap <Double, Example> testSet = new TreeMap <Double, Example>(Collections.reverseOrder());

    // for each example in the training set calculate the cosine simularity and
    // put the corresponding simularity in a tree map. This takes nlog(n) steps
    for (Map.Entry<Example, HashMapVector> m: trainingSet.entrySet()){

      HashMapVector trainingTFIDF = m.getValue();
      // using cosine as the key put the hashvector into the map
      if (trainingTFIDF.length() != 0){
        testSet.put(trainingTFIDF.cosineTo(tdIDFTestExample), m.getKey());
      }
    }

    //take the top k from the testSet
    int count =0;

    //for each category this array stores the frequncy of the
    //feature
    int[] categoriesFreq = new int[categories.length];

    // go through top k elements in the test set
    for (Map.Entry<Double, Example> m: testSet.entrySet()){

      // if count is less than k
      if (count < k){
        // increment count
        count++;
        // for that category increment frequency
        categoriesFreq[m.getValue().category]++;
      }else{
        break;
      }

    }

    // get the maximum frequncy
    int max = 0;

    // category with max frequency
    int categoryToReturn = 0;

    // for loop going through the category frequncy array
    for (int i =0; i < categoriesFreq.length; i++){

        // if max is less than the frequncy, update the category and max
        if (max < categoriesFreq[i]){
          categoryToReturn = i;
          max = categoriesFreq[i];
        }
    }

    // return if the classified category is the same as the real category
    return categoryToReturn == testExample.category;

  }


}
