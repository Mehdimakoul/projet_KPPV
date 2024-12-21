package kppv;

import java.io.*;
import java.util.Arrays;

/**
 * La classe Projet implémente l'algorithme des k plus proches voisins (k-PPV) pour la classification.
 * Elle utilise la méthode du plus proche voisin (1-PPV) ainsi que la méthode des k plus proches voisins (k-PPV).
 * L'algorithme est évalué sur un ensemble de données "iris.data" contenant des exemples de trois classes différentes.
 */
public class Projet {
	
    // Déclaration des constantes
    static int NbEx = 50, NbClasses = 3, NbFeatures = 4, NbExLearning = 25;
    // Les données d'entrée
    static Double data[][][] = new Double[NbClasses][NbEx][NbFeatures];

    // Fonction principale
    public static void main(String[] args) {
        System.out.println("Démarrage de kPPV");
        try {
            // Lecture du fichier de données
            ReadFile();
        } catch (FileNotFoundException e) {
            System.out.println("Le fichier 'iris.data' est introuvable.");
            return;
        }
        // Vérification de la présence de l'argument pour spécifier KNeighbors
        if (args.length < 1) {
            System.out.println("Veuillez spécifier le nombre de voisins (KNeighbors) en argument.");
            return;
        }

        // Récupération du nombre de voisins K pour k-PPV à partir des arguments
        int KNeighbors = Integer.parseInt(args[0]);
        
        Double[] distances = new Double[NbClasses * NbExLearning];

        // X est un exemple à classifier (données de test)
        for (int c = 0; c < NbClasses; c++) {
            for (int i = NbExLearning; i < NbEx; i++) {
                Double[] X = data[c][i];

                // Calcul des distances entre X et tous les exemples d'apprentissage
                ComputeDistances(X, distances);

                
            }}

        

        System.out.println("************************ Evaluation de notre modèle sur l'ensemble de test *************************** ");
        Eval(KNeighbors);

        
        System.out.println("************************ La Cross Validation *************************** ");
        System.out.println(" - L'accuracy donnée par la cross validation (fold =5) : " + crossValidation(KNeighbors, 5) + "\n");
        System.out.println(" - L'accuracy donnée par la cross validation (fold =10): " + crossValidation(KNeighbors, 10) + "\n");
        System.out.println(" - L'accuracy donnée par la cross validation (fold =20): " + crossValidation(KNeighbors, 20) + "\n");
        System.out.println(" - L'accuracy donnée par la cross validation (fold =25): " + crossValidation(KNeighbors, 25) + "\n");
       
    }
     
    /**
     * Calcule les distances entre un exemple X et tous les exemples d'apprentissage.
     * @param x L'exemple à classifier
     * @param distances Le tableau des distances à remplir
     */
    private static void ComputeDistances(Double x[], Double distances[]) {
        int index = 0;
        for (int c = 0; c < NbClasses; c++) {
            for (int n = 0; n < NbExLearning; n++) {
                Double[] example = data[c][n];
                Double distance = 0.0;
                for (int f = 0; f < NbFeatures; f++) {
                    distance += Math.pow(x[f] - example[f], 2);
                }
                distances[index] = Math.sqrt(distance);
                index++;
            }
        }
    }

    /**
     * Recherche la classe d'un point en utilisant la méthode du plus proche voisin (1-PPV).
     * @param distances Les distances entre le point et les exemples d'apprentissage
     * @return La classe prédite pour le point
     */
    private static int findClass(Double distances[]) {
        // Initialisation de la distance minimale et de l'indice correspondant
        double minDistance = distances[0];
        int minIndex = 0;

        // Parcours des distances pour trouver la plus petite
        for (int i = 1; i < distances.length; i++) {
            if (distances[i] < minDistance) {
                minDistance = distances[i];
                minIndex = i;
            }
        }

        // Calcul de l'indice de classe en utilisant le nombre d'exemples par classe
        return minIndex / NbExLearning;
    }

    /**
     * Recherche la classe d'un point en utilisant la méthode des k plus proches voisins (k-PPV).
     * @param distances Les distances entre le point et les exemples d'apprentissage
     * @param KNeighbors Le nombre de voisins à considérer
     * @return La classe prédite pour le point
     */
    private static int PredictClassKNearestNeighbours(Double distances[], int KNeighbors){
    	// Création d'un tableau pour stocker le nombre d'occurrences de chaque classe
        int[] classOccurrences = new int[NbClasses];
        Arrays.fill(classOccurrences, 0);

        // Tri des distances par ordre croissant et sélection des K plus proches voisins
        Double[] sortedDistances = distances.clone();
        Arrays.sort(sortedDistances);
        for (int i = 0; i < KNeighbors; i++) {
            Double currentDistance = sortedDistances[i];
            for (int j = 0; j < distances.length; j++) {
                if (distances[j] == currentDistance) {
                    int currentClass = j / NbExLearning;
                    classOccurrences[currentClass]++;
                    break;
                }
            }
        }

        // Sélection de la classe majoritaire parmi les voisins
        int maxOccurrences = 0;
        int predictedClass = 0;
        for (int i = 0; i < NbClasses; i++) {
            if (classOccurrences[i] > maxOccurrences) {
                maxOccurrences = classOccurrences[i];
                predictedClass = i;
            }
        }

        return predictedClass;
    }


    /**
     * Effectue la validation croisée pour évaluer les performances du modèle.
     * @param k Le nombre de voisins à considérer dans l'algorithme k-PPV
     * @param nFolds Le nombre de plis pour la validation croisée
     * @return L'accuracy moyenne sur tous les plis
     */
    private static double crossValidation(int k, int nFolds) {
        // Initialisation de l'accuracy totale
        double accuracyTotale = 0;
        int taillePlis = NbExLearning / nFolds;

        // Parcours de tous les plis
        for (int fold = 0; fold < nFolds; fold++) {
            int debutTest = fold * taillePlis;
            int finTest = debutTest + taillePlis;
            double correct = 0;

            // Parcours des données de test dans ce pli
            for (int classe = 0; classe < NbClasses; classe++) {
                for (int index = debutTest; index < finTest; index++) {
                    Double[] exemple = data[classe][index];
                    int classeReelle = classe;

                    // Calcul des distances pour l'exemple
                    Double[] distances = new Double[NbClasses * (NbExLearning - taillePlis)];
                    ComputeDistances2(exemple, distances, debutTest, finTest);

                    // Prédiction de la classe de l'exemple en utilisant k-PPV
                    int classePredite = PredictClassKNearestNeighbours(distances, k);

                    // Vérification si la prédiction est correcte
                    if (classePredite == classeReelle) {
                        correct++;
                    }
                }
            }

            // Calcul de l'accuracy pour ce pli
            double accuracyPli = correct / (taillePlis * NbClasses);
            accuracyTotale += accuracyPli;
        }

        // Calcul de l'accuracy moyenne sur tous les plis
        double accuracyMoyenne = accuracyTotale / nFolds;
        return accuracyMoyenne;
    }

   /**
    * Calcule les distances entre un exemple X et chaque élément de l'ensemble d'apprentissage pour la Cross Validation.
    * @param x L'exemple à classifier
    * @param distances Le tableau des distances à remplir
    * @param testStart L'indice de début du pli de test
    * @param testEnd L'indice de fin du pli de test
    */
    private static void ComputeDistances2(Double x[], Double distances[], int testStart, int testEnd) {
        // Initialisation
        int index = 0;

        // Calculer la distance euclidienne entre x et chaque élément de l'ensemble d'apprentissage
        for(int i = 0; i < NbClasses; i++){
            for(int j = 0; j < NbExLearning; j++){
                if (j < testStart || j >= testEnd) {
                    Double distance = 0.0;
                    for (int k = 0; k < NbFeatures; k++) {
                        distance += Math.pow(data[i][j][k] - x[k], 2);
                    }
                    // Ajouter au tableau des distances
                    distances[index] = Math.sqrt(distance);
                    index++;
                }
            }
        }
    }

   
    /**
     * Évalue l'algorithme en calculant la matrice de confusion et le taux de reconnaissance.
     * @param k Le nombre de voisins à considérer
     */
    private static void Eval(int k){
        // Initialisation des variables
        int NbExTesting = NbEx - NbExLearning;
        int prediction[] = new int[NbClasses * NbExTesting];
        int[][] confusionMatrix = new int[NbClasses][NbClasses];
        int correctPredictions = 0;
        int index = 0;
      
        // Calcul de la matrice de confusion et du taux de reconnaissance
        for(int i = 0; i < NbClasses; i++){
            for(int j = NbExLearning; j < NbEx; j++){
                Double X[] = data[i][j];
                Double distances[] = new Double[NbClasses * NbExTesting];

                // Calcul des distances pour l'exemple X
                ComputeDistances(X, distances);

                // Prédiction de la classe de l'exemple X en utilisant les K plus proches voisins
                int X_predicted_class = PredictClassKNearestNeighbours(distances, k);
                prediction[index] = X_predicted_class;
                int real_class = i;

                confusionMatrix[real_class][X_predicted_class]++;
                index++;

                if (X_predicted_class == i) {
                    correctPredictions++;
                }
            }
        }

        // Affichage de la matrice de confusion
        System.out.println("Matrice de confusion :");
        for (int i = 0; i < NbClasses; i++) {
            for (int j = 0; j < NbClasses; j++) {
                System.out.print(confusionMatrix[i][j] + "\t");
            }
            System.out.println();
        }

        // Affichage du taux de reconnaissance
        double recognitionRate = (double) correctPredictions / (NbClasses * (NbEx - NbExLearning));
        System.out.println("Recognition rate: " + recognitionRate );
    }


    /**
     * Lit les données à partir du fichier iris.data.
     * Chaque ligne représente un exemple.
     * Les 50 premières lignes sont 50 exemples de classe 0, les 50 suivantes de classe 1 et les 50 dernières de classe 2.
     * @throws FileNotFoundException Si le fichier iris.data est introuvable
     */
    private static void ReadFile() throws FileNotFoundException {
        String line, subPart;
        int classe = 0, n = 0;
        try (BufferedReader fic = new BufferedReader(new FileReader("iris.data"))) {
            while ((line = fic.readLine()) != null) {
                for (int i = 0; i < NbFeatures; i++) {
                    subPart = line.substring(i * NbFeatures, i * NbFeatures + 3);
                    data[classe][n][i] = Double.parseDouble(subPart);
                }
                if (++n == NbEx) {
                    n = 0;
                    classe++;
                }
            }
        } catch (IOException e) {
            throw new FileNotFoundException("Le fichier 'iris.data' est introuvable.");
        }
    }

} //-------------------Fin de la classe kPPV-------------------------