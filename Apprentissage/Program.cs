using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.MachineLearning;
using Accord.Collections;
using Accord.Neuro;
using Accord.Neuro.Learning;
using Accord.Neuro.Networks;
using Accord.Neuro.ActivationFunctions;



namespace Apprentissage
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // Définir entrée et sortie - XOR
            double[][] inputs = new double[][]
            {
                new double[] { 0 , 0 },
                new double[] { 0 , 1 },
                new double[] { 1 , 0 },
                new double[] { 1 , 1 }
            };

            double[][] outputs = new double[][]
            {
                new double[] { 0 },
                new double[] { 1 },
                new double[] { 1 },
                new double[] { 0 }
            };

            ActivationNetwork network = new ActivationNetwork(
                new SigmoidFunction(2),
                2, // nombre de neurones d'entrée
                2, // nombre de neurones dans la couche cachée
                1  // nombre de neurones dans la couche de sortie
            );

            new NguyenWidrow(network).Randomize();

            BackPropagationLearning teacher = new BackPropagationLearning(network)
            {
                LearningRate = 0.1,
                Momentum = 0.9
            };

            // Entrainer le réseau de neurones
            int iteration = 0;
            double error;
            do
            {
                error = teacher.RunEpoch(inputs, outputs);
                iteration++;
                Console.WriteLine($"Iteration {iteration}, Error: {error}");
            }
            while (error > 0.01);

            foreach (var input in inputs)
            {
                double[] output = network.Compute(input);
                Console.WriteLine($"Input: {input[0]}, {input[1]} - Output: {output[0]}");
            }

            Console.WriteLine("Apprentissage terminé. Appuyez sur une touche pour quitter.");
            Console.ReadKey();
        }
    }
}