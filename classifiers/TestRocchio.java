package ir.classifiers;

import java.util.*;

/**
 * Wrapper class to test NaiveBayes classifier using 10-fold CV.
 * Running it with -debug option gives very detailed output
 *
 * @author Sugato Basu
 */

public class TestRocchio {
  /**
   * A driver method for testing the NaiveBayes classifier using
   * 10-fold cross validation.
   *
   * @param args a list of command-line arguments.  Specifying "-debug"
   *             will provide detailed output
   */
  public static void main(String args[]) throws Exception {
    // REPLACE WITH /u/mooney/ir-code
    String dirName = "/u/mooney/ir-code/corpora/curlie-science/";
    String[] categories = {"bio", "chem", "phys"};
    System.out.println("Loading Examples from " + dirName + "...");
    List<Example> examples = new DirectoryExamplesConstructor(dirName, categories).getExamples();
    System.out.println("Initializing Rocchio classifier...");
    Rocchio rocchio;
    boolean debug;
    // setting debug flag gives very detailed output, suitable for debugging
    if (args.length == 1 && args[0].equals("-debug"))
      debug = true;
    else
      debug = false;

    boolean modified = false;
    for (int i =0; i < args.length; i++){
      if (args[i].equals("-neg")){
        modified = true;
      }
    }
    rocchio = new Rocchio(categories, debug, modified);

    // our example passes all the cases in the small example
    // rocchio.train(examples);
    // Perform 10-fold cross validation to generate learning curve
    CVLearningCurve cvCurve = new CVLearningCurve(rocchio, examples);
    cvCurve.run();
  }
}
