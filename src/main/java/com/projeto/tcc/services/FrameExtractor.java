package com.projeto.tcc.services;


import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
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
public class FrameExtractor {

    @Autowired
    private HOGExtractor hogExtractor;

    @Autowired
    private ORBExtractor orbExtractor;

    @Autowired
    private CNNExtractor cnnExtractor;

    public static String matToString(Mat mat) {
        StringBuilder sb = new StringBuilder();
        // Concatenar número de linhas, colunas e tipo de dados
        sb.append(mat.rows()).append(": ").append(mat.cols()).append(": ").append(mat.type()).append("\n");
        // Concatenar dados do Mat
        ByteBuffer buffer = mat.createBuffer();
        while (buffer.hasRemaining()) {
            byte b = buffer.get();
            sb.append(String.format("%02X", b & 0xFF)).append(" ");
        }
        sb.append("\n");
        return sb.toString();
    }

    public static String indArrayToString(INDArray array) {
        // Convertendo o INDArray para uma string no formato CSV
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < array.rows(); i++) {
            for (int j = 0; j < array.columns(); j++) {
                sb.append(array.getDouble(i, j));
                if (j < array.columns() - 1) {
                    sb.append(",");
                }
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    private Mat resizeFrame(Mat frame, Size size) {
        Mat resizedFrame = new Mat();
        resize(frame, resizedFrame, size);
        return resizedFrame;
    }

    private Mat applyGaussianBlur(Mat frame, Size ksize, double sigmaX) {
        Mat blurredFrame = new Mat();
        GaussianBlur(frame, blurredFrame, ksize, sigmaX);
        return blurredFrame;
    }

    private Mat preprocessFrame(Mat frame) {
        // Redimensiona o frame
        Size newSize = new Size(224, 224); // Exemplo de novo tamanho
        frame = resizeFrame(frame, newSize);

        // Aplica o filtro Gaussiano
        frame = applyGaussianBlur(frame, new Size(9, 9), 1.5);
        return frame;
    }


    public void extractFeatures(String framesPath) throws IOException {
        File dir = new File(framesPath);
        File[] files = dir.listFiles();
        if (files == null) {
            throw new IOException("Não foi possível listar os arquivos no diretório especificado.");
        }
        Arrays.sort(files);

        Scanner scanner = new Scanner(System.in);
        System.out.println("Escolha o extrator de características:");
        System.out.println("1 - HOG");
        System.out.println("2 - ORB");
        System.out.println("3 - CNN");
        System.out.println("4 - Voltar");
        System.out.print("Opção: ");
        int option = scanner.nextInt();

        String extractorName = "";
        switch (option) {
            case 1:
                extractorName = "HOG";
                break;
            case 2:
                extractorName = "ORB";
                break;
            case 3:
                extractorName = "CNN";
                break;
            case 4:
                break;
            default:
                System.out.println("Opção inválida.");
                return;
        }

        String descriptorsPath = "D:\\UFU\\tcc_video_frames\\descriptores\\extracao" + extractorName + "1";
        File descriptorsDir = new File(descriptorsPath);
        if (!descriptorsDir.exists() && !descriptorsDir.mkdirs()) {
            throw new IOException("Não foi possível criar o diretório para os descritores.");
        }

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(descriptorsPath + "\\descritores" + extractorName + ".txt"))) {
            for (File file : files) {
                Mat frame = imread(file.getAbsolutePath());
                frame = preprocessFrame(frame);
                switch (option) {
                    case 1 -> {
                        float[] hogFeatures = hogExtractor.hogExtract(frame);
                        writer.write(Arrays.toString(hogFeatures) + "\n\n");
                    }
                    case 2 -> {
                        Mat orbDescriptors = orbExtractor.orbFeaturesExtractor(frame);
                        writer.write(matToString(orbDescriptors) + "\n\n");
                    }
                    case 3 -> {
                        INDArray cnnFeatures = cnnExtractor.cnnFeaturesExtractor(file.getAbsolutePath());
                        writer.write(indArrayToString(cnnFeatures) + "\n\n");
                    }
                }
            }
        }
    }
}
