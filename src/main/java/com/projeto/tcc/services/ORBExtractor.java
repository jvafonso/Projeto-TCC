package com.projeto.tcc.services;

import org.opencv.core.*;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.ORB;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

@Service
public class ORBExtractor {
    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }
    public static Mat orbFeaturesExtractor(Mat image) {
        // Recebe a imagem já carregada (Imgcodecs.imread)
        // Extrai características ORB das imagem
        ORB orb = ORB.create();
        MatOfKeyPoint keypoints = new MatOfKeyPoint();
        // Criação do Mat que sera retornado e pode ser comparado posteriormente
        Mat descriptors = new Mat();
        orb.detectAndCompute(image, new Mat(), keypoints, descriptors);
        return descriptors;
    }

    public static MatOfDMatch compareFeatures(Mat descriptors1, Mat descriptors2) {
        //uso da distância de Hamming para descriptores binários
        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
        MatOfDMatch matches = new MatOfDMatch();
        matcher.match(descriptors1, descriptors2, matches);

        // Filtragem de correspondências
        List<DMatch> goodMatchesList = filterMatches(matches);
        MatOfDMatch goodMatches = new MatOfDMatch();
        goodMatches.fromList(goodMatchesList);
        // Uso de goodMatches.getTotal() para ter o numero de correspondencias entre as imagens
        return goodMatches;
    }

    // Função para filtrar correspondências com base na distância Hamming definica como multiplo da distancia minima entre as correspondencias
    public static List<DMatch> filterMatches(MatOfDMatch matches) {
        //valores maximo e mínimo
        double maxDist = Double.MIN_VALUE;
        double minDist = Double.MAX_VALUE;
        for (DMatch match : matches.toArray()) {
            double dist = match.distance;
            if (dist < minDist) minDist = dist;
            if (dist > maxDist) maxDist = dist;
        }
        // Limar utilizado para verificar a similaridade entre as imagens
        double threshold = 3 * minDist;
        List<DMatch> goodMatchesList = new ArrayList<>();
        for (DMatch match : matches.toArray()) {
            if (match.distance < threshold) {
                goodMatchesList.add(match);
            }
        }
        return goodMatchesList;
    }
}
