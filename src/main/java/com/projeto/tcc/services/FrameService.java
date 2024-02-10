package com.projeto.tcc.services;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameConverter;
import org.bytedeco.javacv.Java2DFrameUtils;
import org.opencv.core.Core;
import org.opencv.objdetect.CascadeClassifier;
import org.springframework.stereotype.Service;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Scanner;

@Service
@Slf4j
public class FrameService {

    private static final int RGB_PIXEL = 0xff;
    private static final int RGB_RED_BITS = 16;
    private static final int RGB_GREEN_BITS = 8;
    private static final int RGB_COLOR_NUMBER = 3;
    private static final int TOTAL_PIXEL_SIZE = 255;

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
                Frame frame = frameGrabber.grab();
                if (frame == null) {
                    break;
                }
                BufferedImage image = Java2DFrameUtils.toBufferedImage(frame);
                ImageIO.write(image, "png", new File(framesPath + "/video-frame-" + System.currentTimeMillis() + ".png"));
            }
            log.info("Todos os frames foram armazenados");
            frameGrabber.stop();
        } catch (Exception e) {
            log.error("Falha no processo de obtenção dos frames.");
            e.printStackTrace();
        }
    }

    public Double frameComparator(BufferedImage frame1, BufferedImage frame2) {
        try {
            int width1 = frame1.getWidth();
            int width2 = frame2.getWidth();
            int height1 = frame1.getHeight();
            int height2 = frame2.getHeight();

            if ((width1 != width2) || (height1 != height2)) {
                log.error("Frames de dimensões diferentes!");
                throw new IOException();
            } else {
                long frameDiff = 0;

                for (int y = 0; y < height1; y++) {
                    for (int x = 0; x < width1; x++) {
                        int rgb1 = frame1.getRGB(x, y);
                        int rgb2 = frame1.getRGB(x, y);

                        //diferença da cor RGB vermelha
                        frameDiff += Math.abs(((rgb1 >> RGB_RED_BITS) & RGB_PIXEL) - ((rgb2 >> RGB_RED_BITS) & RGB_PIXEL));
                        //diferenca da cor RGB verde
                        frameDiff += Math.abs(((rgb1 >> RGB_GREEN_BITS) & RGB_PIXEL) - ((rgb2 >> RGB_GREEN_BITS) & RGB_PIXEL));
                        //diferenca da cor RGB azul
                        frameDiff += Math.abs(((rgb1) & RGB_PIXEL) - ((rgb2) & RGB_PIXEL));
                    }
                }
                // total de pixels de uma cor RGB = width * height, como são 3 cores
                double totalPixels = width1 * height1 * RGB_COLOR_NUMBER;
                //normalização (médoa) da diferença entre os pixels para acurácia
                double differentPixelsAvg = frameDiff / totalPixels;
                //transformação da diferença entre os frames em um percentual
                return (differentPixelsAvg / TOTAL_PIXEL_SIZE) * 100;
            }

        } catch (Exception e) {
            log.error("Falha no processo de comparação dos frames.");
            e.printStackTrace();
        }
        return null;
    }

}
