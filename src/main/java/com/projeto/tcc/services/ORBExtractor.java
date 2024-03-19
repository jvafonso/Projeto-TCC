package com.projeto.tcc.services;



import lombok.extern.slf4j.Slf4j;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.DMatchVector;
import org.bytedeco.opencv.opencv_features2d.ORB;
import org.bytedeco.opencv.opencv_features2d.DescriptorMatcher;
import org.bytedeco.opencv.opencv_core.DMatch;
import org.springframework.stereotype.Service;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import static org.bytedeco.opencv.global.opencv_core.*;
import org.bytedeco.javacpp.indexer.FloatIndexer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.*;

@Service
@Slf4j
public class ORBExtractor {

    public Mat orbFeaturesExtractor(Mat image) {
        // Recebe a imagem já carregada (Imgcodecs.imread)
        // Extrai características ORB das imagem
        log.info("Extração ORB");
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

    public KeyPointVector orbDetect(Mat frame) {
        // Cria o detector ORB
        ORB orb = ORB.create();
        // Cria um objeto para armazenar os pontos-chave detectados
        KeyPointVector keypoints = new KeyPointVector();
        // Detecta os pontos-chave
        orb.detect(frame, keypoints);

        return keypoints;
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

        // Determinar o tamanho e tipo mais comuns
        Map<Size, Integer> sizeCountMap = new HashMap<>();
        Map<Integer, Integer> typeCountMap = new HashMap<>();
        for (Mat mat : mats) {
            Size size = mat.size();
            sizeCountMap.merge(size, 1, Integer::sum);
            typeCountMap.merge(mat.type(), 1, Integer::sum);
        }
        Size commonSize = Collections.max(sizeCountMap.entrySet(), Map.Entry.comparingByValue()).getKey();
        int commonType = Collections.max(typeCountMap.entrySet(), Map.Entry.comparingByValue()).getKey();

        // Ajustar Mats incompatíveis
        List<Mat> adjustedMats = new ArrayList<>();
        for (Mat mat : mats) {
            if (!mat.size().equals(commonSize) || mat.type() != commonType) {
                Mat adjustedMat = new Mat();
                resize(mat, adjustedMat, new Size((int) commonSize.width(), (int) commonSize.height()));
                adjustedMats.add(adjustedMat);
            } else {
                adjustedMats.add(mat.clone());
            }
        }

        // Calcular a média
        Mat sum = new Mat(commonSize, CV_32F);
        for (Mat mat : adjustedMats) {
            Mat floatMat = new Mat();
            mat.convertTo(floatMat, CV_32F); // Convertendo para float
            add(sum, floatMat, sum);
        }

        // Dividir cada elemento do sum pelo número de Mats
        FloatIndexer sumIndexer = sum.createIndexer();
        for (int y = 0; y < sum.rows(); y++) {
            for (int x = 0; x < sum.cols(); x++) {
                sumIndexer.put(y, x, sumIndexer.get(y, x) / adjustedMats.size());
            }
        }
        sumIndexer.release();

        Mat average = new Mat();
        sum.convertTo(average, commonType); // Convertendo de volta para o tipo comum

        return average;
    }


    public List<Mat> groupFramesORB(File descriptorFile, double similarityThreshold) throws IOException {
        List<List<Mat>> groups = new ArrayList<>();
        List<Mat> selectedDescriptors = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(descriptorFile))) {
            List<String> descriptorLines = new ArrayList<>();
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isEmpty()) {
                    if (!descriptorLines.isEmpty()) {
                        Mat descriptor = stringToMat(String.join("\n", descriptorLines));
                        descriptorLines.clear();
                        boolean addedToGroup = false;
                        for (List<Mat> group : groups) {
                            Mat groupDescriptor = calculateAverageMat(group);
                            DMatchVector matches = compareFeatures(groupDescriptor, descriptor);
                            if (matches.size() < similarityThreshold) {
                                group.add(descriptor);
                                addedToGroup = true;
                                break;
                            }
                        }
                        if (!addedToGroup) {
                            List<Mat> newGroup = new ArrayList<>();
                            newGroup.add(descriptor);
                            groups.add(newGroup);
                        }
                    }
                } else {
                    descriptorLines.add(line);
                }
            }
            // Process the last descriptor
            if (!descriptorLines.isEmpty()) {
                Mat descriptor = stringToMat(String.join("\n", descriptorLines));
                boolean addedToGroup = false;
                for (List<Mat> group : groups) {
                    Mat groupDescriptor = calculateAverageMat(group);
                    DMatchVector matches = compareFeatures(groupDescriptor, descriptor);
                    if (matches.size() < similarityThreshold) {
                        group.add(descriptor);
                        addedToGroup = true;
                        break;
                    }
                }
                if (!addedToGroup) {
                    List<Mat> newGroup = new ArrayList<>();
                    newGroup.add(descriptor);
                    groups.add(newGroup);
                }
            }
        }

        // Create samples from groups
        for (List<Mat> group : groups) {
            Mat groupDescriptor = calculateAverageMat(group);
            Mat selectedDescriptor = null;
            double maxMatches = -1;
            for (Mat descriptor : group) {
                DMatchVector matches = compareFeatures(groupDescriptor, descriptor);
                if (matches.size() > maxMatches) {
                    maxMatches = matches.size();
                    selectedDescriptor = descriptor;
                }
            }
            if (selectedDescriptor != null) {
                selectedDescriptors.add(selectedDescriptor);
            }
        }

        return selectedDescriptors;
    }




}
