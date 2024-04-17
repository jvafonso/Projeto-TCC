package com.projeto.tcc.services;



import lombok.extern.slf4j.Slf4j;
import org.antlr.v4.runtime.atn.SemanticContext;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.DMatchVector;
import org.bytedeco.opencv.opencv_features2d.ORB;
import org.bytedeco.opencv.opencv_features2d.DescriptorMatcher;
import org.bytedeco.opencv.opencv_core.DMatch;
import org.springframework.stereotype.Service;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_core.*;
import org.bytedeco.javacpp.indexer.FloatIndexer;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.*;
import java.util.Arrays;

@Service
@Slf4j
public class ORBExtractor {


    private final ORB orb;

    public ORBExtractor() {
        // Inicializa o ORB com 1000 keypoints
        orb = ORB.create();
        orb.setMaxFeatures(1000);
    }

    public Mat orbFeaturesExtractor(Mat image) {
        KeyPointVector keypoints = new KeyPointVector();
        Mat descriptors = new Mat();
        orb.detectAndCompute(image, new Mat(), keypoints, descriptors);
        if (descriptors.type() != opencv_core.CV_8U) {
            throw new IllegalArgumentException("Os descritores não são do tipo CV_8U.");
        }
        return descriptors;
    }
    public List<Mat> orbFeaturesExtractorBatch(List<Mat> frames) {
        List<Mat> descriptorsList = new ArrayList<>();
        for (Mat frame : frames) {
            descriptorsList.add(orbFeaturesExtractor(frame));
        }
        return descriptorsList;
    }

    /*
    public static double compareFeatures(Mat descriptors1, Mat descriptors2) {
        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
        DMatchVector matches = new DMatchVector();
        matcher.match(descriptors1, descriptors2, matches);

        // Filtragem de correspondências
        DMatchVector goodMatches = filterMatches(matches);

        // Normaliza a medida de similaridade para o intervalo [0.1, 1]
        double similarity = goodMatches.size() / (double) matches.size();

        return 0.9 * similarity + 0.1;
    }
     */

    // Função para filtrar correspondências com base na distância Hamming definica como multiplo da distancia minima entre as correspondencias
    public static DMatchVector  filterMatches(DMatchVector  matches) {
        //valores maximo e mínimo
        double maxDist = Double.MIN_VALUE;
        double minDist = Double.MAX_VALUE;
        for (int i = 0; i < matches.size(); i++) {
            double dist = matches.get(i).distance();
            if (dist < minDist) minDist = dist;
            if (dist > maxDist) maxDist = dist;
        }
        // Limar utilizado para verificar a similaridade entre as imagens
        log.info("Filtragem ORB");
        double threshold = 1.5 * minDist;
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

    public static double compareFeatures(Mat descriptors1, Mat descriptors2) {
        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
        DMatchVector matches = new DMatchVector();
        int maxMatches = 50;
        matcher.match(descriptors1, descriptors2, matches);

        // Calcular a média das distâncias de Hamming dos melhores matches
        double distanciaHammingMedia = 0;
        int count = (int) Math.min(maxMatches, matches.size());
        for (int i = 0; i < count; i++) {
            distanciaHammingMedia += matches.get(i).distance();
        }
        distanciaHammingMedia /= count;

        // Normalizar a distância média para o intervalo [0, 1]
        double maxDistancia = descriptors1.cols() * 8; // Número máximo de bits diferentes
        double distanciaNormalizada = distanciaHammingMedia / maxDistancia;

        // Converter a distância normalizada em uma medida de similaridade
        double similaridade = 1 - distanciaNormalizada;

        return similaridade;
    }


    public static Mat stringToMat(String str) {
        String[] parts = str.split("\n", 2);
        String[] header = parts[0].split(": ");
        int rows = Integer.parseInt(header[0].trim());
        int cols = Integer.parseInt(header[1].trim());
        int type = Integer.parseInt(header[2].trim());
        Mat mat = new Mat(rows, cols, type);
        ByteBuffer buffer = mat.createBuffer();
        String[] byteStrings = parts[1].trim().split(" ");
        for (String byteString : byteStrings) {
            if (!byteString.isEmpty()) {
                byte b = (byte) Integer.parseInt(byteString, 16);
                buffer.put(b);
            }
        }
        return mat;
    }

    public static Mat calculateAverageMat(List<Mat> mats) {
        if (mats == null || mats.isEmpty()) {
            throw new IllegalArgumentException("A lista não pode ser vazia.");
        }

        // Determinar o tamanho comum
        Size commonSize = mats.get(0).size();

        // Calcular a média
        Mat sum = new Mat(commonSize, CV_32F, new Scalar(0)); // Inicializa a matriz de soma com zeros
        Mat resizedMat = new Mat();
        Mat floatMat = new Mat();
        for (Mat mat : mats) {
            resize(mat, resizedMat, commonSize); // Redimensiona o Mat para o tamanho comum
            resizedMat.convertTo(floatMat, CV_32F); // Conversão para float
            add(sum, floatMat, sum); // Adição in-place
        }

        // Criar um Mat representando o escalar (número de mats)
        Mat divisor = new Mat(commonSize, CV_32F, new Scalar(mats.size()));
        divide(sum, divisor, sum); // Divisão in-place

        // Convertendo de volta para o tipo comum
        Mat average = new Mat();
        sum.convertTo(average, mats.get(0).type());

        // Liberação de recursos
        sum.release();
        resizedMat.release();
        floatMat.release();
        divisor.release();

        return average;
    }


    public List<Mat> groupFramesORB(File descriptorFile, double similarityThreshold, double samplingPercentage) throws IOException {
        List<ORBGroup> groups = new ArrayList<>();
        List<Mat> selectedDescriptors = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(descriptorFile))) {
            List<String> descriptorLines = new ArrayList<>();
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isEmpty()) {
                    if (!descriptorLines.isEmpty()) {
                        Mat descriptor = stringToMat(String.join("\n", descriptorLines));
                        descriptorLines.clear();
                        addToORBGroup(groups, descriptor, similarityThreshold);
                    }
                } else {
                    descriptorLines.add(line);
                }
            }
            // Process the last descriptor
            if (!descriptorLines.isEmpty()) {
                Mat descriptor = stringToMat(String.join("\n", descriptorLines));
                addToORBGroup(groups, descriptor, similarityThreshold);
            }
        }

        // Create samples from groups
        for (ORBGroup group : groups) {
            Mat selectedDescriptor = group.selectRepresentative();
            if (selectedDescriptor != null) {
                selectedDescriptors.add(selectedDescriptor);
            }
            log.info("Quadro adicionado a amostra!!");
        }

        // Sample the selected descriptors
        int elementsToSample = (int) (selectedDescriptors.size() * (samplingPercentage / 100.0));
        if (elementsToSample == 0 && !selectedDescriptors.isEmpty()) {
            elementsToSample = 1; // Garante que pelo menos um elemento seja selecionado
        }
        return selectedDescriptors.subList(0, elementsToSample);
    }

    private void addToORBGroup(List<ORBGroup> groups, Mat descriptor, double similarityThreshold) {
        for (ORBGroup group : groups) {
            if (group.isSimilar(descriptor, similarityThreshold)) {
                group.add(descriptor);
                log.info("Quadro adicionado ao grupo");
                return;
            }
        }
        log.info("Novo grupo criado");
        groups.add(new ORBGroup(descriptor));
    }

    private static class ORBGroup {
        private List<Mat> descriptors = new ArrayList<>();
        private Mat average;

        ORBGroup(Mat descriptor) {
            add(descriptor);
        }

        void add(Mat descriptor) {
            descriptors.add(descriptor);
            updateAverage();
        }

        boolean isSimilar(Mat descriptor, double similarityThreshold) {
            if (average == null) {
                return true;
            }
            double similarity = compareFeatures(average, descriptor);
            log.info("Distancia: {}, similarityThreshold: {}", similarity, similarityThreshold);
            return similarity < similarityThreshold;
        }

        Mat selectRepresentative() {
            Mat selectedDescriptor = null;
            double maxSimilarity = -1;
            for (Mat descriptor : descriptors) {
                double similarity = compareFeatures(average, descriptor);
                if (similarity > maxSimilarity) {
                    maxSimilarity = similarity;
                    selectedDescriptor = descriptor;
                }
            }
            return selectedDescriptor;
        }

        private void updateAverage() {
            average = calculateAverageMat(descriptors);
        }
    }

}
