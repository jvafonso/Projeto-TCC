package com.projeto.tcc.services;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameUtils;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

import org.bytedeco.opencv.opencv_core.DMatchVector;
import org.bytedeco.opencv.opencv_core.Mat;
import org.nd4j.linalg.api.ndarray.INDArray;


import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;



@Service
@Slf4j
public class FrameService {

    @Autowired
    private HOGExtractor hogExtractor;

    @Autowired
    private ORBExtractor orbExtractor;

    @Autowired
    private CNNExtractor cnnExtractor;

    public void samplingFrames() throws IOException, Exception {
        Scanner sc = new Scanner(System.in);
        System.out.print("Insira o caminho do arquivo de vídeo de entrada: ");
        String videoPath = sc.nextLine();

        System.out.print("Insira o caminho em que os frames serão armazenados: ");
        String framesPath = sc.nextLine();

        FFmpegFrameGrabber frameGrabber = new FFmpegFrameGrabber(videoPath);
        frameGrabber.start();
        try{
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
            extractAndCompareFeatures(framesPath);
        } catch (Exception e) {
            log.error("Falha no processo de obtenção dos frames.");
            e.printStackTrace();
        }
    }

    public void extractAndCompareFeatures(String framesPath) throws IOException {
        File dir = new File(framesPath);
        File[] files = dir.listFiles();
        Arrays.sort(files);

        float[] previousHOGFeatures = null;
        Mat previousORBDescriptors = null;
        INDArray previousCNNFeatures = null;

        for (int i = 1; i < files.length; i++) {
            File currentFile = files[i];
            File previousFile = files[i - 1];

            Mat currentFrame = imread(currentFile.getAbsolutePath());

            // HOG
            float[] currentHOGFeatures = hogExtractor.hogExtract(currentFrame);
            if (previousHOGFeatures != null) {
                double hogDistance = hogExtractor.distanciaEuclidiana(previousHOGFeatures, currentHOGFeatures);
                System.out.println("HOG: Distância Euclidiana entre frame " + (i - 1) + " e frame " + i + ": " + hogDistance);
            }
            previousHOGFeatures = currentHOGFeatures;

            // ORB
            Mat currentORBDescriptors = orbExtractor.orbFeaturesExtractor(currentFrame);
            if (previousORBDescriptors != null) {
                // Verificar se as matrizes de descritores têm o mesmo número de colunas
                if (previousORBDescriptors.cols() != currentORBDescriptors.cols()) {
                    log.warn("O número de colunas dos descritores não é igual.");
                } else {
                    DMatchVector orbMatches = orbExtractor.compareFeatures(previousORBDescriptors, currentORBDescriptors);
                    System.out.println("ORB: Número de correspondências entre frame " + (i - 1) + " e frame " + i + ": " + orbMatches.size());
                }
            }
            previousORBDescriptors = currentORBDescriptors;

            // CNN
            INDArray currentCNNFeatures = cnnExtractor.cnnFeaturesExtractor(currentFile.getAbsolutePath());
            if (previousCNNFeatures != null) {
                double cnnSimilarity = cnnExtractor.compareFeatures(previousCNNFeatures, currentCNNFeatures);
                System.out.println("CNN: Similaridade entre frame " + (i - 1) + " e frame " + i + ": " + cnnSimilarity);
            }
            previousCNNFeatures = currentCNNFeatures;

            if (i == files.length - 1) {
                System.out.println("Não há mais imagens para comparar. O programa será encerrado.");
                return;
            }
        }

    }

}
