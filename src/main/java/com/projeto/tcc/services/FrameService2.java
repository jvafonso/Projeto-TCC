package com.projeto.tcc.services;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameUtils;
import org.bytedeco.opencv.opencv_core.DMatchVector;
import org.bytedeco.opencv.opencv_core.KeyPointVector;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.nd4j.linalg.api.ndarray.INDArray;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

import org.springframework.beans.factory.annotation.Autowired;
import org.bytedeco.opencv.opencv_core.KeyPoint;
import org.springframework.stereotype.Service;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.ByteBuffer;
import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import org.bytedeco.javacpp.indexer.UByteIndexer;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.opencv_core.Scalar;
@Service
@Slf4j
public class FrameService2 {

    @Autowired
    private HOGExtractor hogExtractor;

    @Autowired
    private ORBExtractor orbExtractor;

    @Autowired
    private CNNExtractor cnnExtractor;

    private final Lock groupsLock = new ReentrantLock();

    public void samplingFrames(String videoPath, double similarityThreshold) throws IOException, FFmpegFrameGrabber.Exception {
        long startTime = System.currentTimeMillis();

        String videoName = new File(videoPath).getName();
        String framesPath = "D:\\UFU\\tcc_video_frames\\frames\\" + videoName;

        File framesDir = new File(framesPath);
        if (!framesDir.exists() && !framesDir.mkdirs()) {
            throw new IOException("Não foi possível criar o diretório para armazenar os frames.");
        }

        FFmpegFrameGrabber frameGrabber = new FFmpegFrameGrabber(videoPath);
        frameGrabber.start();
        try {
            for (int i = 0; i < frameGrabber.getLengthInFrames(); i++) {
                Frame frame = frameGrabber.grabImage();
                if (frame == null) {
                    break;
                }
                BufferedImage image = Java2DFrameUtils.toBufferedImage(frame);
                ImageIO.write(image, "png", new File(framesPath + "/video-frame-" + System.currentTimeMillis() + ".png"));
            }
            log.info("Todos os frames foram armazenados");
            frameGrabber.stop();
            long endTime = System.currentTimeMillis();
            log.info("Tempo de execução da separação dos frames: " + (endTime - startTime) + " ms");
        } catch (IOException e) {
            log.error("Falha no processo de obtenção dos frames.");
            e.printStackTrace();
        }
    }



    public void groupFrames(String descriptorsFilePath, double similarityThreshold) throws IOException {
        File file = new File(descriptorsFilePath);
        String fileName = file.getName();
        String extractorName = "";
        List<?> sample = Collections.emptyList();

        if (fileName.contains("HOG")) {
            sample = hogExtractor.groupFramesHog(file, similarityThreshold);
            extractorName = "HOG";
        } else if (fileName.contains("ORB")) {
            sample = orbExtractor.groupFramesORB(file, similarityThreshold);
            extractorName = "ORB";
        } else if (fileName.contains("CNN")) {
            sample = cnnExtractor.groupFramesCNN(file, similarityThreshold);
            extractorName = "CNN";
        } else {
            throw new IllegalArgumentException("Tipo de descritor desconhecido.");
        }

        if (!sample.isEmpty()) {
            saveAsPointsFile(sample, "D:\\UFU\\tcc_video_frames\\amostras\\extracao" + extractorName + "\\pointsFile", extractorName);
        }
    }


    private void saveAsPointsFile(Object descriptors, String filePath, String extractorName) throws IOException {
        int dimensionality = 0;
        List<String> pointLines = new ArrayList<>();
        if (descriptors instanceof List<?> descriptorList) {
            if (!descriptorList.isEmpty()) {
                if (descriptorList.get(0) instanceof float[]) {
                    dimensionality = ((float[]) descriptorList.get(0)).length;
                    for (Object descriptorObj : descriptorList) {
                        float[] descriptor = (float[]) descriptorObj;
                        StringBuilder sb = new StringBuilder();
                        for (int i = 0; i < dimensionality; i++) {
                            sb.append(descriptor[i]).append(i < dimensionality - 1 ? ";" : "");
                        }
                        pointLines.add(sb.toString());
                    }
                } else if (descriptorList.get(0) instanceof Mat) {
                    dimensionality = ((Mat) descriptorList.get(0)).cols();
                    for (Object descriptorObj : descriptorList) {
                        Mat descriptor = (Mat) descriptorObj;
                        StringBuilder sb = new StringBuilder();
                        for (int i = 0; i < dimensionality; i++) {
                            sb.append(descriptor.getFloatBuffer().get(i)).append(i < dimensionality - 1 ? ";" : "");
                        }
                        pointLines.add(sb.toString());
                    }
                } else if (descriptorList.get(0) instanceof INDArray) {
                    dimensionality = (int) ((INDArray) descriptorList.get(0)).length();
                    for (Object descriptorObj : descriptorList) {
                        INDArray descriptor = (INDArray) descriptorObj;
                        StringBuilder sb = new StringBuilder();
                        for (int i = 0; i < dimensionality; i++) {
                            sb.append(descriptor.getDouble(i)).append(i < dimensionality - 1 ? ";" : "");
                        }
                        pointLines.add(sb.toString());
                    }
                }
            }
        }

        if (!pointLines.isEmpty()) {
            File baseDir = new File(filePath.substring(0, filePath.lastIndexOf(File.separator)));
            if (!baseDir.exists()) {
                baseDir.mkdirs();
            }

            int fileIndex = 1;
            String fileName = filePath + fileIndex + ".data"; // Alteração da extensão para .data
            while (new File(fileName).exists()) {
                fileIndex++;
                fileName = filePath + fileIndex + ".data"; // Alteração da extensão para .data
            }

            try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileName))) {
                writer.write("DN\n"); // Representação densa sem classe
                writer.write(pointLines.size() + "\n"); // Número de pontos
                writer.write(dimensionality + "\n"); // Dimensionalidade dos pontos
                for (int i = 0; i < dimensionality; i++) {
                    writer.write("D" + i + (i < dimensionality - 1 ? ";" : "\n"));
                }
                for (String line : pointLines) {
                    writer.write("P" + pointLines.indexOf(line) + ";" + line + "\n");
                }
            }
        }
    }
}
