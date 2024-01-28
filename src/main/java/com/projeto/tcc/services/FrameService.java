package com.projeto.tcc.services;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameConverter;
import org.bytedeco.javacv.Java2DFrameUtils;
import org.springframework.stereotype.Service;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Scanner;

@Service
@Slf4j
public class FrameService {

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
}
