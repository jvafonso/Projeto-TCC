package com.projeto.tcc.services;




import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.opencv.opencv_core.Size;
import org.opencv.core.MatOfFloat;
import org.bytedeco.opencv.global.opencv_objdetect;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_objdetect.HOGDescriptor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

@Service
@Slf4j
public class HOGExtractor {
    @Value("${opencv.native.library.path}")
    private String opencvNativeLibraryPath;

    public float[] hogExtract(Mat frame) {
        //Já recebe a imagem carregada (Imgcodecs.imread)
        //Criação do objeto HOGDescriptor
        log.info("Extração HOG");
        HOGDescriptor hog = new HOGDescriptor();
        FloatPointer descriptor = new FloatPointer();
        hog.compute(frame, descriptor);
        // Normalização das características HOG
        /*
        float sum = 0;
        for (int i = 0; i < descriptor.limit(); i++) {
            sum += descriptor.get(i) * descriptor.get(i);
        }
        float norm = (float) Math.sqrt(sum);
        for (int i = 0; i < descriptor.limit(); i++) {
            descriptor.put(i, descriptor.get(i) / norm);
        }

         */
        // Converte o descritor para vetor para facilitar a comparação e retorna
        int descriptorSize = (int) (hog.getDescriptorSize());
        float[] descriptorArray = new float[descriptorSize];
        descriptor.get(descriptorArray);
        //extracao das caracteristicas do frame
        log.info("Fim da Extração HOG");
        return descriptorArray;
    }

    public double distanciaEuclidiana(float[] vector1, float[] vector2) {
        double distancia = 0;
        for (int i = 0; i < vector1.length; i++) {
            distancia += Math.pow(vector1[i] - vector2[i], 2);
        }
        return Math.sqrt(distancia);
    }
}
