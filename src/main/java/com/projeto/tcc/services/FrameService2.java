package com.projeto.tcc.services;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.nd4j.linalg.api.ndarray.INDArray;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

import org.springframework.beans.factory.annotation.Autowired;
import org.bytedeco.opencv.opencv_core.KeyPoint;
import org.springframework.stereotype.Service;
import java.io.*;
import java.nio.ByteBuffer;
import java.util.*;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
@Service
@Slf4j
public class FrameService2 {

    @Autowired
    private HOGExtractor hogExtractor;

    @Autowired
    private ORBExtractor orbExtractor;

    @Autowired
    private CNNExtractor cnnExtractor;

    @Autowired
    PointsFileGenerator pointsFileGenerator;

    private final Lock groupsLock = new ReentrantLock();
    private static final int BATCH_SIZE = 30;

    private String DEFAULT_PATH = "D:\\UFU\\tcc_video_frames\\amostras\\extracao";


    public void groupFrames(String descriptorsFilePath, double similarityThreshold, double samplingPercentage) throws IOException {
        File file = new File(descriptorsFilePath);
        String fileName = file.getName();
        String extractorName = "";
        List<?> sample = Collections.emptyList();

        if (fileName.contains("HOG")) {
            log.info("Agrupamento HOG");
            List<float[]> sampleHOG = hogExtractor.groupFramesHog(file, similarityThreshold, samplingPercentage);
            log.info("Fim agrupamento HOG, {}", sampleHOG.isEmpty());
            extractorName = "HOG";
            pointsFileGenerator.saveAsPointsFileHOG(sampleHOG,DEFAULT_PATH + extractorName + "\\pointsFile");
        } else if (fileName.contains("ORB")) {
            log.info("Agrupamento ORB");
            List<Mat> sampleORB = orbExtractor.groupFramesORB(file, similarityThreshold, samplingPercentage);
            log.info("Fim agrupamento ORB, {}", sampleORB.isEmpty());
            extractorName = "ORB";
            pointsFileGenerator.saveAsPointsFileORB(sampleORB,DEFAULT_PATH + extractorName + "\\pointsFile");
        } else if (fileName.contains("CNN")) {
            log.info("Agrupamento CNN");
            List<INDArray> sampleCNN = cnnExtractor.groupFramesCNN(file, similarityThreshold, samplingPercentage);
            log.info("Fim agrupamento CNN, {}", sampleCNN.isEmpty());
            extractorName = "CNN";
            pointsFileGenerator.saveAsPointsFileCNN(sampleCNN,DEFAULT_PATH + extractorName + "\\pointsFile");
        } else {
            throw new IllegalArgumentException("Tipo de descritor desconhecido.");
        }
    }


    public void pointsFileHoleSet(File file, String filePath, String extractorName) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            List<Object> descriptors = new ArrayList<>();
            List<String> descriptorLines = new ArrayList<>();
            String line;
            StringBuilder descriptorString = new StringBuilder();
            while ((line = reader.readLine()) != null) {
                if (extractorName.equals("HOG")) {
                    if (line.trim().isEmpty()) {
                        if (!descriptorLines.isEmpty()) {
                            descriptors.add(hogExtractor.parseDescriptor(descriptorLines));
                            descriptorLines.clear();
                        }
                    } else {
                        descriptorLines.add(line);
                    }
                } else if (extractorName.equals("ORB")) {
                    if (line.isEmpty()) {
                        if (!descriptorLines.isEmpty()) {
                            descriptors.add(ORBExtractor.stringToMat(String.join("\n", descriptorLines)));
                            descriptorLines.clear();
                        }
                    } else {
                        descriptorLines.add(line);
                    }
                } else if (extractorName.equals("CNN")) {
                    if (line.trim().isEmpty()) {
                        if (descriptorString.length() > 0) {
                            descriptors.add(CNNExtractor.stringToINDArray(descriptorString.toString()));
                            descriptorString.setLength(0); // Clear the StringBuilder
                        }
                    } else {
                        descriptorString.append(line).append("\n");
                    }
                }
            }

            // Salva todos os descritores em um único arquivo Points File
            if (!descriptors.isEmpty()) {
                switch (extractorName) {
                    case "HOG" -> {
                        List<float[]> hogBatch = new ArrayList<>();
                        for (Object obj : descriptors) {
                            hogBatch.add((float[]) obj);
                        }
                        pointsFileGenerator.saveAsPointsFileHOG(hogBatch, filePath);
                    }
                    case "ORB" -> {
                        List<Mat> orbBatch = new ArrayList<>();
                        for (Object obj : descriptors) {
                            orbBatch.add((Mat) obj);
                        }
                        pointsFileGenerator.saveAsPointsFileORB(orbBatch, filePath);
                    }
                    case "CNN" -> {
                        List<INDArray> cnnBatch = new ArrayList<>();
                        for (Object obj : descriptors) {
                            cnnBatch.add((INDArray) obj);
                        }
                        pointsFileGenerator.saveAsPointsFileCNN(cnnBatch, filePath);
                    }
                }
            }
        }
    }

    public void randomSample(String descriptorsFilePath, double samplingPercentage) throws IOException {
        File file = new File(descriptorsFilePath);
        String fileName = file.getName();
        String extractorName = "";

        if (fileName.contains("HOG")) {
            log.info("Agrupamento HOG");
            List<float[]> sampleHOG = hogExtractor.sampleRandomDescriptors(file, samplingPercentage);
            log.info("Fim agrupamento HOG, {}", sampleHOG.isEmpty());
            extractorName = "HOG";
            pointsFileGenerator.saveAsPointsFileHOG(sampleHOG,DEFAULT_PATH + extractorName + "\\pointsFile");
        }  else if (fileName.contains("CNN")) {
            log.info("Agrupamento CNN");
            List<INDArray> sampleCNN = cnnExtractor.sampleRandomCNNDescriptors(file, samplingPercentage);
            log.info("Fim agrupamento CNN, {}", sampleCNN.isEmpty());
            extractorName = "CNN";
            pointsFileGenerator.saveAsPointsFileCNN(sampleCNN,DEFAULT_PATH + extractorName + "\\pointsFile");
        } else {
            throw new IllegalArgumentException("Tipo de descritor desconhecido.");
        }

    }

    public void sampleBySecond(String descriptorsFilePath, double samplingPercentage) throws IOException {
        File file = new File(descriptorsFilePath);
        String fileName = file.getName();
        String extractorName = "";

        if (fileName.contains("HOG")) {
            log.info("Agrupamento HOG");
            List<float[]> sampleHOG = hogExtractor.sampleFramesBySecondHOG(file, samplingPercentage);
            log.info("Fim agrupamento HOG, {}", sampleHOG.isEmpty());
            extractorName = "HOG";
            pointsFileGenerator.saveAsPointsFileHOG(sampleHOG,DEFAULT_PATH + extractorName + "\\pointsFile");
        }  else if (fileName.contains("CNN")) {
            log.info("Agrupamento CNN");
            List<INDArray> sampleCNN = cnnExtractor.sampleFramesBySecondCNN(file, samplingPercentage);
            log.info("Fim agrupamento CNN, {}", sampleCNN.isEmpty());
            extractorName = "CNN";
            pointsFileGenerator.saveAsPointsFileCNN(sampleCNN,DEFAULT_PATH + extractorName + "\\pointsFile");
        } else {
            throw new IllegalArgumentException("Tipo de descritor desconhecido.");
        }

    }

}
