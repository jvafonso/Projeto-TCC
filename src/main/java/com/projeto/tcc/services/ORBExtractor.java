package com.projeto.tcc.services;



import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.opencv_core.KeyPointVector;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.DMatchVector;
import org.bytedeco.opencv.opencv_features2d.ORB;
import org.bytedeco.opencv.global.opencv_features2d;
import org.bytedeco.opencv.opencv_features2d.DescriptorMatcher;
import org.bytedeco.opencv.opencv_core.DMatch;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

@Service
@Slf4j
public class ORBExtractor {

    public Mat orbFeaturesExtractor(Mat image) {
        // Recebe a imagem já carregada (Imgcodecs.imread)
        // Extrai características ORB das imagem
        ORB orb = ORB.create();
        KeyPointVector keypoints = new KeyPointVector();
        // Criação do Mat que sera retornado e pode ser comparado posteriormente
        Mat descriptors = new Mat();
        orb.detectAndCompute(image, new Mat(), keypoints, descriptors);
        // Verifica se o tipo dos descritores é CV_8U
        if (descriptors.type() != opencv_core.CV_8U) {
            throw new IllegalArgumentException("Os descritores não são do tipo CV_8U.");
        }
        return descriptors;
    }

    public DMatchVector compareFeatures(Mat descriptors1, Mat descriptors2) {
        //uso da distância de Hamming para descriptores binários
        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
        DMatchVector matches = new DMatchVector();
        matcher.match(descriptors1, descriptors2, matches);

        // Filtragem de correspondências
        DMatchVector goodMatches = filterMatches(matches);
        // Uso de goodMatches.getTotal() para ter o numero de correspondencias entre as imagens
        return goodMatches;
    }

    // Função para filtrar correspondências com base na distância Hamming definica como multiplo da distancia minima entre as correspondencias
    public DMatchVector  filterMatches(DMatchVector  matches) {
        //valores maximo e mínimo
        double maxDist = Double.MIN_VALUE;
        double minDist = Double.MAX_VALUE;
        for (int i = 0; i < matches.size(); i++) {
            double dist = matches.get(i).distance();
            if (dist < minDist) minDist = dist;
            if (dist > maxDist) maxDist = dist;
        }
        // Limar utilizado para verificar a similaridade entre as imagens
        double threshold = 3 * minDist;
        List<DMatch> goodMatchesList  = new ArrayList<>();
        for (int i = 0; i < matches.size(); i++) {
            if (matches.get(i).distance() < threshold) {
                goodMatchesList .add(matches.get(i));
            }
        }
        // Converte a lista de índices para um DMatchVector
        DMatchVector goodMatches = new DMatchVector(goodMatchesList.size());
        for (int i = 0; i < goodMatchesList.size(); i++) {
            goodMatches.put(i, goodMatchesList.get(i));;
        }
        return goodMatches;
    }
}
