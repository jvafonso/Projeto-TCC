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
import java.io.*;
import java.util.*;
import java.util.concurrent.*;

@Service
@Slf4j
public class FrameExtractor {

    @Autowired
    private HOGExtractor hogExtractor;

    @Autowired
    private ORBExtractor orbExtractor;

    @Autowired
    private CNNExtractor cnnExtractor;

    private static final int BATCH_SIZE = 30;

    public void samplingFrames(String videoPath) throws IOException, FFmpegFrameGrabber.Exception {
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


    public void extractFeatures(String framesPath, String videoName) throws IOException {
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
                return;
            default:
                System.out.println("Opção inválida.");
                return;
        }

        File baseDir = new File("D:\\UFU\\tcc_video_frames\\descriptores");
        int dirIndex = 1;
        File descriptorsDir;
        do {
            descriptorsDir = new File(baseDir, videoName + "extracao" + extractorName + dirIndex);
            dirIndex++;
        } while (descriptorsDir.exists());

        if (!descriptorsDir.mkdirs()) {
            throw new IOException("Não foi possível criar o diretório para os descritores.");
        }

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File(descriptorsDir, "descritores" + extractorName + ".txt")))) {
            List<Mat> frames = new ArrayList<>();
            List<String> imagePaths = new ArrayList<>();
            for (File file : files) {
                if (option == 3) {
                    // Para CNN, adiciona o caminho da imagem
                    imagePaths.add(file.getAbsolutePath());
                } else {
                    // Para HOG e ORB, carrega o frame e pré-processa
                    Mat frame = imread(file.getAbsolutePath());
                    frame = preprocessFrame(frame);
                    frames.add(frame);
                }
            }

            // Processa os quadros em lote e escreve os descritores
            switch (option) {
                case 1:
                    List<float[]> hogFeaturesList = hogExtractor.hogExtractBatch(frames);
                    for (float[] hogFeatures : hogFeaturesList) {
                        writer.write(Arrays.toString(hogFeatures) + "\n\n");
                    }
                    break;
                case 2:
                    List<Mat> orbDescriptorsList = orbExtractor.orbFeaturesExtractorBatch(frames);
                    for (Mat orbDescriptors : orbDescriptorsList) {
                        writer.write(matToString(orbDescriptors) + "\n\n");
                    }
                    break;
                case 3:
                    List<INDArray> cnnFeaturesList = cnnExtractor.cnnFeaturesExtractorBatch(imagePaths);
                    for (INDArray cnnFeatures : cnnFeaturesList) {
                        writer.write(indArrayToString(cnnFeatures) + "\n\n");
                    }
                    break;
            }
        }
    }

    public void extractFeaturesV2(String framesPath, String videoName) throws IOException, InterruptedException, ExecutionException {
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
                return;
            default:
                System.out.println("Opção inválida.");
                return;
        }

        File baseDir = new File("D:\\UFU\\tcc_video_frames\\descriptores");
        int dirIndex = 1;
        File descriptorsDir;
        do {
            descriptorsDir = new File(baseDir, videoName + "extracao" + extractorName + dirIndex);
            dirIndex++;
        } while (descriptorsDir.exists());

        if (!descriptorsDir.mkdirs()) {
            throw new IOException("Não foi possível criar o diretório para os descritores.");
        }

        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        List<Callable<Void>> tasks = new ArrayList<>();

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File(descriptorsDir, "descritores" + extractorName + ".txt")))) {
            // Subdivide a lista de arquivos em lotes para processamento paralelo
            for (int i = 0; i < files.length; i += BATCH_SIZE) {
                final int start = i;
                final int end = Math.min(files.length, i + BATCH_SIZE);

                tasks.add(() -> {
                    List<Mat> batchFrames = new ArrayList<>();
                    List<String> batchImagePaths = new ArrayList<>();

                    for (int j = start; j < end; j++) {
                        if (option == 3) {
                            batchImagePaths.add(files[j].getAbsolutePath());
                        } else {
                            Mat frame = imread(files[j].getAbsolutePath());
                            frame = preprocessFrame(frame);
                            batchFrames.add(frame);
                        }
                    }

                    switch (option) {
                        case 1:
                            List<float[]> hogFeaturesBatch = hogExtractor.hogExtractBatch(batchFrames);
                            synchronized (writer) {
                                for (float[] hogFeatures : hogFeaturesBatch) {
                                    writer.write(Arrays.toString(hogFeatures) + "\n\n");
                                }
                            }
                            break;
                        case 2:
                            List<Mat> orbDescriptorsBatch = orbExtractor.orbFeaturesExtractorBatch(batchFrames);
                            synchronized (writer) {
                                for (Mat orbDescriptors : orbDescriptorsBatch) {
                                    writer.write(matToString(orbDescriptors) + "\n\n");
                                }
                            }
                            break;
                        case 3:
                            List<INDArray> cnnFeaturesBatch = cnnExtractor.cnnFeaturesExtractorBatch(batchImagePaths);
                            synchronized (writer) {
                                for (INDArray cnnFeatures : cnnFeaturesBatch) {
                                    writer.write(indArrayToString(cnnFeatures) + "\n\n");
                                }
                            }
                            break;
                    }
                    return null;
                });
            }

            List<Future<Void>> futures = executor.invokeAll(tasks);

            // Espera todas as tarefas serem concluídas
            for (Future<Void> future : futures) {
                future.get();
            }
        } finally {
            executor.shutdown();
        }
    }

}
