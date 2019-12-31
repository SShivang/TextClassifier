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



public class Rocchio extends Classifier {

  /**
   * Trains the classifier on the training examples
   *
   * @param trainingExamples a list of Example objects that will be used
   *                         for training the classifier
   */

  // inverted index object for the training examples
  private InvertedIndex index = null;

  // flag to use the modified rocchio
  private boolean modified;

  // name of the classifier
  public static String name = "Rocchio";

  // a map of the prototypeVectors
  private HashMap<Integer,HashMapVector> prototypeVectors;

  // debug flag
  private boolean debug;

  /*
    Constructor for the Rocchio class.
  */
  public Rocchio(String[] categories, boolean debug, boolean modified) {
    this.categories = categories;
    this.debug = debug;
    this.modified = modified;
    if (modified){
      name += "Modified";
    }
    prototypeVectors = new HashMap<Integer,HashMapVector>();
  }

  /* This method returns the name of the classifier */
  public String getName(){
    return name;
  }

  /*
    This method is designed to train the Rocchio algorithm. It takes
    in a set of training examples. It sets the tf-idf of the training
    samples. If the tf-idf has already been calculated for the example
    it doesn't recalculate for the example.
  */
  public void train(List<Example> trainingExamples){

    // Initialize the prototype vector hashmap
    prototypeVectors = new HashMap<Integer,HashMapVector>();

    // make an inverted index object to get the idfs
    index = new InvertedIndex(trainingExamples);

    // Initialize the prototype vectors
    for (int i =0; i < categories.length; i++)
      prototypeVectors.put(i, new HashMapVector());

    // for each example
    for (Example e : trainingExamples){

      // fetch the prototype vector and the vector for the example
      HashMapVector prototype = prototypeVectors.get(e.category);
      HashMapVector h = e.hashVector;

      // calculate the tfidf vector
      HashMapVector tfIdf = tfIdfVector(h, e);

      // add the tf-idf vector to the prototype
      // the scaling is done inside the tfIdf function
      prototype.add(tfIdf);


      // if the modified rocchio metric subtract the prototype from all
      // the other prototype vectors
      if(modified){
        for (int i =0; i < categories.length; i++){
          if (i != e.category){
            prototype = prototypeVectors.get(i);
            prototype.addScaled(tfIdf, -1);
          }
        }
      }

    }

    // passes al the cases in the small examples piazza post.
    // prototype vectors are corrct
    // System.out.println(prototypeVectors);

  }

  /*
    This method calculates the tf idf vector for a given sample
    in place for space efficiency.
  */

  public HashMapVector tfIdfVector(HashMapVector original, Example e){

    // copy the original vector
    HashMapVector vector = original.copy();

    // load the map of idf
    Map<String, TokenInfo>  map = index.tokenHash;

    // get the max weight
    double maxWeight = vector.maxWeight();

    // for each entry
    for (Map.Entry<String, Weight> entry : vector.hashMap.entrySet()) {

      // weight of the entry
      Weight weight = entry.getValue();

      // get the token
      String token = entry.getKey();

      // set the weight to the idf, if its not in the map it isn't in the prototype
      if (map.get(token) != null){
        weight.setValue( map.get(token).idf * weight.getValue()/ maxWeight);
      }

    }

    return vector;

  }


  /**
   * Returns true if the predicted category of the test example matches the correct category,
   * false otherwise
   */
  public boolean test(Example testExample){

    // fetch the tf idf vector corresponding to the test example
    HashMapVector tfidf = tfIdfVector(testExample.hashVector, testExample);

    // set m to -2 impossible cosine simularity
    double m = -2;

    // category is set to -1
    int category = -1;

    for (Map.Entry<Integer, HashMapVector> entry : prototypeVectors.entrySet()) {
      // The weight for the token is in the value of the Weight
      HashMapVector hashMap = entry.getValue();

      // calculate simularity
      double simularity = tfidf.cosineTo(hashMap);

      // let m be the largest simularity and category being the category
      // with the largest simularity
      if (m < simularity ){
        category = entry.getKey();
        m = simularity;
      }


    }

    // if not found then use a random category
    if (m == -2){
      Random r = new Random();
		  category = r.nextInt((2 - 0) + 1) + 0;
    }

    // if the category is the same as the example category return true
    return testExample.category == category;
  }


}
