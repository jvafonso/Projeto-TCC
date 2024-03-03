package com.projeto.tcc.services;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameUtils;
import org.bytedeco.opencv.opencv_core.DMatchVector;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.nd4j.linalg.api.ndarray.INDArray;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
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

    private List<File> representativeFrames = new CopyOnWriteArrayList<>();

    private final Lock groupsLock = new ReentrantLock();

    public void samplingFrames() throws IOException, FFmpegFrameGrabber.Exception {
        Scanner sc = new Scanner(System.in);
        System.out.print("Insira o caminho do arquivo de vídeo de entrada: ");
        String videoPath = sc.nextLine();

        System.out.print("Insira o caminho em que os frames serão armazenados: ");
        String framesPath = sc.nextLine();

        System.out.print("Insira a taxa de similaridade (0-100): ");
        double similarityThreshold = sc.nextDouble();
        similarityThreshold /= 100;

        if (similarityThreshold < 0 || similarityThreshold > 1) {
            throw new IllegalArgumentException("A taxa de similaridade deve estar entre 0 e 100.");
        }

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
            groupFrames(framesPath, similarityThreshold);
        } catch (IOException e) {
            log.error("Falha no processo de obtenção dos frames.");
            e.printStackTrace();
        }
    }

    public void groupFrames(String framesPath, double similarityThreshold) throws IOException {
        File dir = new File(framesPath);
        File[] files = dir.listFiles();
        if (files == null) {
            throw new IOException("Não foi possível listar os arquivos no diretório especificado.");
        }
        Arrays.sort(files);

        List<List<File>> groups = new ArrayList<>();

        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        representativeFrames.clear();
        for (int i = 0; i < files.length; i++) {
            final int index = i;
            executor.submit(() -> {
                try {
                    File currentFile = files[index];
                    Mat currentFrame = imread(currentFile.getAbsolutePath());
                    Mat frame = preprocessFrame(currentFrame);

                    float[] currentHOGFeatures = hogExtractor.hogExtract(frame);
                    Mat currentORBDescriptors = orbExtractor.orbFeaturesExtractor(frame);
                    INDArray currentCNNFeatures = cnnExtractor.cnnFeaturesExtractor(currentFile.getAbsolutePath());

                    double minDifference = Double.MAX_VALUE;
                    int minGroupIndex = -1;

                    for (int j = 0; j < representativeFrames.size(); j++) {
                        File representativeFile = representativeFrames.get(j);
                        Mat representativeFrame = imread(representativeFile.getAbsolutePath());

                        float[] representativeHOGFeatures = hogExtractor.hogExtract(representativeFrame);
                        Mat representativeORBDescriptors = orbExtractor.orbFeaturesExtractor(representativeFrame);
                        INDArray representativeCNNFeatures = cnnExtractor.cnnFeaturesExtractor(representativeFile.getAbsolutePath());

                        double hogDistance = hogExtractor.distanciaEuclidiana(representativeHOGFeatures, currentHOGFeatures);
                        DMatchVector orbMatches = orbExtractor.compareFeatures(representativeORBDescriptors, currentORBDescriptors);
                        double cnnSimilarity = cnnExtractor.compareFeatures(representativeCNNFeatures, currentCNNFeatures);

                        double similarityScore = calculateSimilarityScore(hogDistance, orbMatches.size(), cnnSimilarity);

                        double difference = Math.abs(similarityScore - similarityThreshold);
                        log.info("similarityThreshold: [{}], similarityScore: [{}], difference: [{}]", similarityThreshold, similarityScore, difference);
                        if (similarityScore < similarityThreshold && difference < minDifference) {
                            minDifference = difference;
                            minGroupIndex = j;
                        }
                    }

                    groupsLock.lock();
                    try {
                        if (minGroupIndex != -1) {
                            if (minGroupIndex < groups.size()) {
                                groups.get(minGroupIndex).add(currentFile);
                                reelectRepresentativeFrame(groups.get(minGroupIndex), minGroupIndex);
                            } else {
                                log.warn("Índice de grupo inválido: " + minGroupIndex);
                            }

                        } else {
                            List<File> newGroup = new ArrayList<>();
                            newGroup.add(currentFile);
                            groups.add(newGroup);
                            representativeFrames.add(currentFile);
                        }

                    } finally {
                        groupsLock.unlock();
                    }

                } catch (Exception e) {
                    log.error("Erro ao processar o frame " + index, e);
                }
            });
        }

        executor.shutdown();
        try {
            if (!executor.awaitTermination(1, TimeUnit.HOURS)) {
                log.error("O tempo de execução excedeu o limite.");
                executor.shutdownNow(); // Força o encerramento das tarefas restantes
            }
        } catch (InterruptedException e) {
            log.error("A execução foi interrompida.", e);
            Thread.currentThread().interrupt();
            executor.shutdownNow();
        }

        log.info("Número de grupos criados: " + groups.size());
        saveGroups(groups, framesPath);
        saveRepresentativeFrames(representativeFrames, framesPath);
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

    private double calculateSimilarityScore(double hogDistance, Long orbMatches, double cnnSimilarity) {
        double hogWeight = 0.1; // Peso para a distância HOG
        double orbWeight = 0.3; // Peso para o número de correspondências ORB
        double cnnWeight = 0.6; // Peso para a similaridade CNN

        // Normaliza os valores para que estejam na mesma escala
        double normalizedHogDistance = 1 / (1 + hogDistance); // Quanto menor a distância, maior a similaridade
        double normalizedOrbMatches = orbMatches.doubleValue() / 100.0; // Supondo que o número máximo de correspondências possíveis seja 100
        double normalizedCnnSimilarity = cnnSimilarity / 100.0; // Supondo que a similaridade máxima de CNN seja 100

        // Calcular a pontuação de similaridade como a média ponderada dos valores normalizados
        double similarityScore = (hogWeight * normalizedHogDistance) + (orbWeight * normalizedOrbMatches) + (cnnWeight * normalizedCnnSimilarity);

        return similarityScore;
    }

    private void reelectRepresentativeFrame(List<File> group, int groupIndex) {
        double minAverageDistance = Double.MAX_VALUE;
        File newRepresentativeFrame = null;

        for (File candidateFrame : group) {
            double totalDistance = 0;
            for (File otherFrame : group) {
                if (!candidateFrame.equals(otherFrame)) {
                    totalDistance += calculateDistance(candidateFrame, otherFrame);
                }
            }
            double averageDistance = totalDistance / (group.size() - 1);
            if (averageDistance < minAverageDistance) {
                minAverageDistance = averageDistance;
                newRepresentativeFrame = candidateFrame;
            }
        }

        if (newRepresentativeFrame != null) {
            representativeFrames.set(groupIndex, newRepresentativeFrame);
        }
    }

    private double calculateDistance(File frame1File, File frame2File) {
        try {
            // Converte as imagens em matrizes do OpenCV
            Mat mat1 = imread(frame1File.getAbsolutePath());
            Mat mat2 = imread(frame2File.getAbsolutePath());

            // Extrai os descritores HOG dos frames
            float[] descriptor1 = hogExtractor.hogExtract(mat1);
            float[] descriptor2 = hogExtractor.hogExtract(mat2);

            // Calcula a distância euclidiana entre os descritores HOG
            return hogExtractor.distanciaEuclidiana(descriptor1, descriptor2);
        } catch (Exception e) {
            e.printStackTrace();
            return Double.MAX_VALUE;
        }
    }

    private void saveGroups(List<List<File>> groups, String framesPath) throws IOException {
        for (int i = 0; i < groups.size(); i++) {
            String groupPath = framesPath + "/grupo" + (i + 1);
            File groupDir = new File(groupPath);
            if (!groupDir.exists() && !groupDir.mkdirs()) {
                throw new IOException("Não foi possível criar o diretório para o grupo " + (i + 1));
            }

            List<File> group = groups.get(i);
            for (File file : group) {
                BufferedImage image = ImageIO.read(file);
                ImageIO.write(image, "png", new File(groupPath + "/" + file.getName()));
            }
        }
    }

    private void saveRepresentativeFrames(List<File> representativeFrames, String framesPath) throws IOException {
        String samplePath = framesPath + "/amostra";
        File sampleDir = new File(samplePath);
        if (!sampleDir.exists() && !sampleDir.mkdirs()) {
            throw new IOException("Não foi possível criar o diretório para armazenar os frames representativos.");
        }

        for (int i = 0; i < representativeFrames.size(); i++) {
            File file = representativeFrames.get(i);
            BufferedImage image = ImageIO.read(file);
            ImageIO.write(image, "jpeg", new File(samplePath + "/representative-frame-" + (i + 1) + ".jpeg"));
        }

        saveAsPointsFile(representativeFrames, samplePath + "/representative-frames.data");
        saveAsDistanceMatrixFile(representativeFrames, samplePath + "/representative-frames.dmat");
    }

    private void saveAsPointsFile(List<File> frames, String filePath) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            writer.write(frames.size() + "\n");
            for (int i = 0; i < frames.size(); i++) {
                writer.write(i + " " + (i * 2) + " " + (i * 3) + "\n");
            }
        }
    }

    private void saveAsDistanceMatrixFile(List<File> frames, String filePath) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            writer.write(frames.size() + "\n");
            for (int i = 0; i < frames.size(); i++) {
                for (int j = 0; j < frames.size(); j++) {
                    double distance = Math.sqrt(Math.pow(i - j, 2) + Math.pow((i * 2) - (j * 2), 2) + Math.pow((i * 3) - (j * 3), 2));
                    writer.write(distance + " ");
                }
                writer.newLine();
            }
        }
    }
}
